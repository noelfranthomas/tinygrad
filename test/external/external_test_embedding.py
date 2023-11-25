from tinygrad.tensor import Tensor
from tinygrad.nn import Embedding
from tinygrad.helpers import GlobalCounters

# OPT=100 NOOPT=1 GRAPHUOPS=1 GRAPH=1 DEBUG=4 python3 test/external/external_test_embedding.py
if __name__ == "__main__":
  vocab_size = 192
  dim = 128
  idx = Tensor([[1,2,3,4,5,6,7]]).realize()
  embed = Embedding(vocab_size, dim)
  embed.weight = Tensor.rand(*embed.weight.shape).realize()
  GlobalCounters.reset()
  ret = embed(idx).numpy()
