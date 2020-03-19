import torch

def square_root_symmetric_matrix(A):
    w, v = torch.symeig(A, eigenvectors=True)
    return torch.mm(torch.mm(v, torch.diag(torch.sqrt(w[:]))), v.t())

if __name__ == '__main__':
    A = [[2,0.],[0.,3]]
    At = torch.tensor(A)
    Bt = square_root_symmetric_matrix(At)
    Ct = torch.mm(Bt,Bt.t())

    print(At)
    print(Bt)
    print(Ct)
