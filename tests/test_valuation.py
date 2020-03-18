import datetime
import torch
from quantitative_analytics.indices import indices, indexfixingrepository
from quantitative_analytics.marketdata import marketdata, marketdatarepository
from quantitative_analytics.products import products
from quantitative_analytics.models import models
from quantitative_analytics.calculators import calculators

torch.set_printoptions(precision=16)

EPSILON = 0.00000001


observationDates = [datetime.date(year=2020, month=11, day=30),
                    datetime.date(year=2021, month=6, day=30),
                    datetime.date(year=2021, month=12, day=30)]
expiry = datetime.date(year=2021, month=12, day=30)
equity = indices.EquityIndex([], "SPX")

fixingDate = datetime.date(year=2020, month=11, day=30)
spot_fixing = torch.tensor(100.0, requires_grad=True)
indexfixingrepository.indexFixingRepositorySingleton.storeFixing(equity, fixingDate, spot_fixing)

forward = torch.tensor([100.0], requires_grad=True)
md = marketdata.MarketDataEquitySpotBase(equity, forward)
marketdatarepository.marketDataRepositorySingleton.storeMarketData(md)

volatilityDates = [datetime.date(year=2021, month=6, day=30),
                    datetime.date(year=2021, month=12, day=30)]
volatilityValues = torch.tensor([0.2,0.2], requires_grad=True)
volatilityMarketData = marketdata.BlackVolatilityMarketData(equity, volatilityDates, volatilityValues)
marketdatarepository.marketDataRepositorySingleton.storeMarketData(volatilityMarketData)

option_data = {}
option_data['strike'] = torch.tensor(100.0, requires_grad=True)
option_data['expiry'] = expiry
option_data['index'] = equity

# Create a European Option
europeanOption = products.EuropeanOptionProduct(option_data)

option_data['observationDates'] = observationDates

# Create an Asian Option
asianOption = products.AsianOptionProduct(option_data)

modelData = {}
modelData['forward'] = forward
modelData['volatility'] = torch.tensor([0.2], requires_grad=True)

modelDate = datetime.date(year=2020, month=12, day=30)
model = models.LognormalModel(modelData, modelDate)


def test_european_option_analytic():

    data = []
    eoc = calculators.EuropeanOptionCalculator(data, model, europeanOption)
    npv = eoc.npv()

    expected_result = torch.tensor(7.9655609130859375)

    assert abs(npv[0] - expected_result) < EPSILON

simulationData = {}
simulationData['NumberOfSimulations'] = 100000

def test_asian_option_monte_carlo():
    # Run the Monte-Carlo
    mc = calculators.MonteCarloSimulator(simulationData, model, asianOption)
    npvmc = mc.npv()

    # Compute first order derivatives
    x = [forward]

    dxs = []
    for it in npvmc:
        dx, = torch.autograd.grad(it, x, create_graph=True, retain_graph=True)
        dxs.append(dx)

    ddxs = []
    for it in dxs:
        ddx, = torch.autograd.grad(it, x, create_graph=True)
        ddxs.append(ddx)

    expected_results = [torch.tensor(100.0135192871093750),torch.tensor(4.2101149559020996)]
    expected_deltas = [torch.tensor(0.6668019294738770),torch.tensor(0.3529491424560547)]
    expected_gammas = [torch.tensor(0.0), torch.tensor(0.0141420941799879)]

    for i,it in enumerate(npvmc):
        assert abs(it - expected_results[i]) < EPSILON

    for i, it in enumerate(dxs):
        assert abs(it - expected_deltas[i]) < EPSILON

    for i, it in enumerate(ddxs):
        assert abs(it - expected_gammas[i]) < EPSILON