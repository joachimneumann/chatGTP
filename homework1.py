# Value class starter code, with many functions taken out
from math import exp, log
import numpy as np

class Value:
  
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0.0
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op
    self.label = label

  def __repr__(self):
    return f"{self.data} grad={self.grad})"
  
  def __add__(self, other): # exactly as in the video
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')    
    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward
    return out
  
  def __radd__(self, other): # other + self
      return self + other
  def __rmul__(self, other): # other + self
      return self * other

  def __mul__(self, other):
      other = other if isinstance(other, Value) else Value(other)
      out = Value(self.data * other.data, (self, other), '*')

      def _backward():
          self.grad += other.data * out.grad
          other.grad += self.data * out.grad
      out._backward = _backward

      return out
  def __truediv__(self, other): # self / other
      other = other if isinstance(other, Value) else Value(other)
      out = Value(self.data / other.data, (self, other), f'/')

      def _backward():
          self.grad +=  out.grad / other.data
          other.grad += self.data * -other.data ** -2 * out.grad
      out._backward = _backward
      return out

  def exp(self):
      x = self.data
      t = np.exp(x)
      out = Value(t, (self, ), 'exp')
      def _backward():
          self.grad = t * out.grad
      out._backward = _backward
      return out
        
  def log(self):
      x = self.data
      t = np.log(x)
      out = Value(t, (self, ), 'log')
      def _backward():
          self.grad = (1.0 / self.data) * out.grad
      out._backward = _backward
      return out

  def __neg__(self): # -self
      return self * -1

  def backward(self): # exactly as in video  
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)
    
    self.grad = 1.0
    for node in reversed(topo):
      node._backward()
