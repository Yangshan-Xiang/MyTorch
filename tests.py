import unittest

from mytorch.models import *
from mytorch.optimizers import *
from mytorch.arithmetics import *

class TestFunctions(unittest.TestCase):

    def setUp(self):
        self.pts = [[1, 2], [3, 4]]
        self.linear_1 = Linear(2, 1)
        self.linear_2 = Linear(2, 2)
        self.linear_3 = Linear(2, 3)
        self.x = Parameter(Tensor([10]))
        self.y = Parameter(Tensor([10]))
        self.z = 10
        self.a = self.x * self.y
        self.b = self.y * self.z
        self.c = self.a + self.b
        self.output = self.c.log()


        self.storage = [1, 2, 3, 4, 5, 6]
        self.shape = (2, 2) # Only need 4 elements in the storage
        self.stride = (1, 2) # Transposed
        self.offset = 2 # Starting from the 3rd element in the storage, skipping the first two
        self.tensor = Tensor(self.storage, self.shape, self.stride, self.offset) # Representing [[3, 5], [4, 6]]

        self.storage2 = [1, 2, 3, 4]
        self.shape2 = (2, 1, 2)
        self.stride2 = (2, 2, 1)
        self.offset2 = 0
        self.tensor2 = Tensor(self.storage2, self.shape2, self.stride2, self.offset2)

        self.storage3 = [1, 2, 3, 4, 5, 6]
        self.shape3 = (3, 2)
        self.stride3 = (1, 3)
        self.offset3 = 0
        self.tensor3 = Tensor(self.storage3, self.shape3, self.stride3, self.offset3)

        self.storage4 = [1, 2, 3, 4, 5, 6]
        self.shape4 = (3, 2)
        self.stride4 = (2, 1)
        self.offset4 = 0
        self.tensor4 = Tensor(self.storage4, self.shape4, self.stride4, self.offset4)

    def test_topological_sort(self):
        right_ordering = [id(self.output), id(self.c), id(self.b), id(self.a), id(self.y), id(self.x)]
        ordering = []

        for i in self.output.topological_sort():
            ordering.append(id(i))

        self.assertEqual(ordering, right_ordering)

    def test_chain_rule(self):
        self.assertEqual(len(self.output.chain_rule(Tensor([1]))), 1)
        self.assertEqual(id(self.output.chain_rule(Tensor([1]))[0][0]), id(self.c))
        self.assertAlmostEqual(self.output.chain_rule(Tensor([1]))[0][1], 1 / 200)

        self.assertEqual(len(self.c.chain_rule(Tensor([1 / 200]))), 2)
        self.assertEqual(id(self.c.chain_rule(Tensor([1 / 200]))[0][0]), id(self.a))
        self.assertEqual(id(self.c.chain_rule(Tensor([1 / 200]))[1][0]), id(self.b))
        self.assertEqual(self.c.chain_rule(Tensor([1 / 200]))[0][1], 1 / 200)
        self.assertEqual(self.c.chain_rule(Tensor([1 / 200]))[1][1], 1 / 200)

    def test_forward(self):
        self.assertAlmostEqual(self.output.storage[0], math.log(200))

    def test_backward(self):
        self.output.backward()

        self.assertAlmostEqual(self.x.gradient.storage[0], 0.05)
        self.assertAlmostEqual(self.y.gradient.storage[0], 0.1)

    def test_Linear(self):
        for pt in self.pts:
            self.assertEqual(len(self.linear_1(Tensor(pt)).storage), 1)
            self.assertEqual(len(self.linear_2(Tensor(pt)).storage), 2)
            self.assertEqual(len(self.linear_3(Tensor(pt)).storage), 3)

    def test_get_params(self):
        self.assertEqual(len(self.linear_1.get_params()[0].storage), 2)
        self.assertEqual(len(self.linear_2.get_params()[0].storage), 4)
        self.assertEqual(len(self.linear_3.get_params()[0].storage), 6)

        self.assertEqual(len(self.linear_1.get_params()[1].storage), 1)
        self.assertEqual(len(self.linear_2.get_params()[1].storage), 2)
        self.assertEqual(len(self.linear_3.get_params()[1].storage), 3)

    def test_SGD(self):
        self.assertIsNone(self.x.gradient)
        self.x.gradient = Tensor([2])
        lr = 0.01
        SGD([self.x], lr).step()
        self.assertEqual(self.x.storage[0], 10 - lr * self.x.gradient)

    def test_Adam(self):
        self.assertIsNone(self.x.gradient)
        self.x.gradient = Tensor([2])
        lr = 0.01

        Adam([self.x], lr).step()
        self.assertEqual(self.x.storage[0], 10 - lr * 2 / (2 + 1e-8))

    def test_Adagrad(self):
        self.assertIsNone(self.x.gradient)
        self.x.gradient = Tensor([2])
        lr = 0.1
        Adagrad([self.x], lr).step()
        self.assertEqual(self.x.storage[0], 10 - lr * 2 / (2 + 1e-8))  # As you can see, it is the same as Adam

    def test_zero_grad(self):
        self.assertIsNone(self.x.gradient)
        self.assertIsNone(self.y.gradient)
        self.x.derivative = Tensor([2])
        self.x.derivative = Tensor([4])

        Adam([self.x, self.y]).zero_grad()

        self.assertIsNone(self.x.gradient)
        self.assertIsNone(self.y.gradient)

    def test_to_storage_idx(self):
        self.assertEqual(to_storage_idx((0, 0), self.stride, self.offset), 2)
        self.assertEqual(to_storage_idx((0, 1), self.stride, self.offset), 4)
        self.assertEqual(to_storage_idx((1, 0), self.stride, self.offset), 3)
        self.assertEqual(to_storage_idx((1, 1), self.stride, self.offset), 5)

    def test_to_tensor_idx(self):
        self.assertEqual(to_tensor_idx(0, self.shape), (0, 0))
        self.assertEqual(to_tensor_idx(1, self.shape), (0, 1))
        self.assertEqual(to_tensor_idx(2, self.shape), (1, 0))
        self.assertEqual(to_tensor_idx(3, self.shape), (1, 1))

    def test_from_broadcast_idx(self):
        self.assertEqual(from_broadcast_idx((0, 0, 0), self.shape), (0, 0))
        self.assertEqual(from_broadcast_idx((0, 0, 1), self.shape), (0, 1))
        self.assertEqual(from_broadcast_idx((0, 1, 0), self.shape), (1, 0))
        self.assertEqual(from_broadcast_idx((0, 1, 1), self.shape), (1, 1))
        self.assertEqual(from_broadcast_idx((1, 0, 0), self.shape), (0, 0))
        self.assertEqual(from_broadcast_idx((1, 0, 1), self.shape), (0, 1))
        self.assertEqual(from_broadcast_idx((1, 1, 0), self.shape), (1, 0))
        self.assertEqual(from_broadcast_idx((1, 1, 1), self.shape), (1, 1))

    def test_is_contiguous(self):
        self.assertEqual(self.tensor.is_contiguous(), False)
        self.assertEqual(self.tensor2.is_contiguous(), True)
        self.assertEqual(self.tensor3.is_contiguous(), False)
        self.assertEqual(self.tensor4.is_contiguous(), True)

    def test_contiguous_stride(self):
        self.assertEqual(self.tensor.contiguous_stride(), (2, 1))
        self.assertEqual(self.tensor2.contiguous_stride(), (2, 2, 1))
        self.assertEqual(self.tensor3.contiguous_stride(), (2, 1))
        self.assertEqual(self.tensor4.contiguous_stride(), (2, 1))

    def test_to_contiguous(self):
        self.tensor.to_contiguous()
        self.assertEqual(self.tensor.storage, [3, 5, 4, 6])
        self.assertEqual(self.tensor.shape, (2, 2))
        self.assertEqual(self.tensor.stride, (2, 1))
        self.assertEqual(self.tensor.offset, 0)

    def test_broadcastable(self):
        self.assertEqual(self.tensor.broadcastable(self.tensor2), (2, 2, 2))
        self.assertEqual(self.tensor.broadcastable(self.tensor3), False)
        self.assertEqual(self.tensor.broadcastable(self.tensor4), False)
        self.assertEqual(self.tensor2.broadcastable(self.tensor3), (2, 3, 2))
        self.assertEqual(self.tensor2.broadcastable(self.tensor4), (2, 3, 2))
        self.assertEqual(self.tensor3.broadcastable(self.tensor4), (3, 2))

    def test_repr(self):
        self.assertEqual(self.tensor.__repr__(), "Tensor([[3, 5], [4, 6]])")
        self.assertEqual(self.tensor2.__repr__(), "Tensor([[[1, 2]], [[3, 4]]])")
        self.assertEqual(self.tensor3.__repr__(), "Tensor([[1, 4], [2, 5], [3, 6]])")
        self.assertEqual(self.tensor4.__repr__(), "Tensor([[1, 2], [3, 4], [5, 6]])")

    def test_neg(self):
        out = -self.tensor
        self.assertEqual(out.storage, [-3, -5, -4, -6])
        self.assertEqual(out.shape, (2, 2))
        self.assertEqual(out.stride, (2, 1))
        self.assertEqual(out.offset, 0)

        out = -self.tensor2
        self.assertEqual(out.storage, [-1, -2, -3, -4])
        self.assertEqual(out.shape, (2, 1, 2))
        self.assertEqual(out.stride, (2, 2, 1))
        self.assertEqual(out.offset, 0)

        out = -self.tensor3
        self.assertEqual(out.storage, [-1, -4, -2, -5, -3, -6])
        self.assertEqual(out.shape, (3, 2))
        self.assertEqual(out.stride, (2, 1))
        self.assertEqual(out.offset, 0)

        out = -self.tensor4
        self.assertEqual(out.storage, [-1, -2, -3, -4, -5, -6])
        self.assertEqual(out.shape, (3, 2))
        self.assertEqual(out.stride, (2, 1))
        self.assertEqual(out.offset, 0)

    def test_add(self):
        out = self.tensor + self.tensor2
        self.assertEqual(out.storage, [4, 7, 5, 8, 6, 9, 7, 10])
        self.assertEqual(out.shape, (2, 2, 2))
        self.assertEqual(out.stride, (4, 2, 1))
        self.assertEqual(out.offset, 0)

        out = self.tensor2 + self.tensor3
        self.assertEqual(out.storage, [2, 6, 3, 7, 4, 8, 4, 8, 5, 9, 6, 10])
        self.assertEqual(out.shape, (2, 3, 2))
        self.assertEqual(out.stride, (6, 2, 1))
        self.assertEqual(out.offset, 0)

        out = self.tensor2 + self.tensor4
        self.assertEqual(out.storage, [2, 4, 4, 6, 6, 8, 4, 6, 6, 8, 8, 10])
        self.assertEqual(out.shape, (2, 3, 2))
        self.assertEqual(out.stride, (6, 2, 1))
        self.assertEqual(out.offset, 0)

        out = self.tensor3 + self.tensor4
        self.assertEqual(out.storage, [2, 6, 5, 9, 8, 12])
        self.assertEqual(out.shape, (3, 2))
        self.assertEqual(out.stride, (2, 1))
        self.assertEqual(out.offset, 0)


        self.assertEqual((self.tensor + 1).__repr__(), "Tensor([[4, 6], [5, 7]])")
        self.assertEqual((1 + self.tensor).__repr__(), "Tensor([[4, 6], [5, 7]])")


        self.assertRaises(ValueError, self.tensor.__add__, self.tensor3)
        self.assertRaises(ValueError, self.tensor.__add__, self.tensor4)

    def test_sub(self):
        self.assertEqual((self.tensor - 1).__repr__(), "Tensor([[2, 4], [3, 5]])")
        self.assertEqual((1 - self.tensor).__repr__(), "Tensor([[-2, -4], [-3, -5]])")

    def test_mul(self):
        self.assertEqual((self.tensor * 2).__repr__(), "Tensor([[6, 10], [8, 12]])")
        self.assertEqual((2 * self.tensor).__repr__(), "Tensor([[6, 10], [8, 12]])")

    def test_div(self):
        self.assertEqual((self.tensor / 2).__repr__(), "Tensor([[1.5, 2.5], [2.0, 3.0]])")
        self.assertEqual((60 / self.tensor).__repr__(), "Tensor([[20.0, 12.0], [15.0, 10.0]])")

    def test_pow(self):
        self.assertEqual((self.tensor ** 2).__repr__(), "Tensor([[9, 25], [16, 36]])") # type: ignore
        self.assertEqual((2 ** self.tensor).__repr__(), "Tensor([[8, 32], [16, 64]])") # type: ignore

    def test_softmax(self):
        out = self.tensor.softmax(0)
        for i, element in enumerate(out.storage):
            self.assertAlmostEqual(element, [0.2689, 0.2689, 0.7311, 0.7311][i], places=4)
        self.assertEqual(out.shape, (2, 2))
        self.assertEqual(out.stride, (2, 1))
        self.assertEqual(out.offset, 0)

        out = self.tensor.softmax(1)
        for i, element in enumerate(out.storage):
            self.assertAlmostEqual(element, [0.1192, 0.8808, 0.1192, 0.8808][i], places=4)
        self.assertEqual(out.shape, (2, 2))
        self.assertEqual(out.stride, (2, 1))
        self.assertEqual(out.offset, 0)

        out = self.tensor4.softmax(0)
        for i, element in enumerate(out.storage):
            self.assertAlmostEqual(element, [0.0159, 0.0159, 0.1173, 0.1173, 0.8668, 0.8668][i], places=4)
        self.assertEqual(out.shape, (3, 2))
        self.assertEqual(out.stride, (2, 1))
        self.assertEqual(out.offset, 0)

        out = self.tensor4.softmax(1)
        for i, element in enumerate(out.storage):
            self.assertAlmostEqual(element, [0.2689, 0.7311, 0.2689, 0.7311, 0.2689, 0.7311][i], places=4)
        self.assertEqual(out.shape, (3, 2))
        self.assertEqual(out.stride, (2, 1))
        self.assertEqual(out.offset, 0)

    def test_sigmoid(self):
        out = self.tensor.sigmoid()
        for i, element in enumerate(out.storage):
            self.assertAlmostEqual(element, [0.9526, 0.9933, 0.982, 0.9975][i], places=4)

        out = self.tensor4.sigmoid()
        for i, element in enumerate(out.storage):
            self.assertAlmostEqual(element, [0.7311, 0.8808, 0.9526, 0.982, 0.9933, 0.9975][i], places=4)

    def test_reshape(self):
        self.assertEqual(self.tensor.reshape(4).__repr__(), "Tensor([3, 5, 4, 6])")
        self.assertEqual(self.tensor2.reshape(2, 2).__repr__(), "Tensor([[1, 2], [3, 4]])")
        self.assertEqual(self.tensor3.reshape(2, 3).__repr__(), "Tensor([[1, 4, 2], [5, 3, 6]])")
        self.assertEqual(self.tensor4.reshape(6,).__repr__(), "Tensor([1, 2, 3, 4, 5, 6])")

    def test_permute(self):
        self.assertEqual(self.tensor.permute(1, 0).__repr__(), "Tensor([[3, 4], [5, 6]])")
        self.assertEqual(self.tensor2.permute(2, 1, 0).__repr__(), "Tensor([[[1, 3]], [[2, 4]]])")
        self.assertEqual(self.tensor3.permute(1, 0).__repr__(), "Tensor([[1, 2, 3], [4, 5, 6]])")
        self.assertEqual(self.tensor4.permute(1, 0).__repr__(), "Tensor([[1, 3, 5], [2, 4, 6]])")

    def test_sum(self):
        self.assertEqual(self.tensor.sum(0).__repr__(), "Tensor([[7, 11]])")
        self.assertEqual(self.tensor.sum(1).__repr__(), "Tensor([[8], [10]])")
        self.assertEqual(self.tensor.sum(0, keepdim=False).__repr__(), "Tensor([7, 11])")
        self.assertEqual(self.tensor.sum(1, keepdim=False).__repr__(), "Tensor([8, 10])")

    def test_matmul(self):
        self.assertEqual((self.tensor @ self.tensor).__repr__(), "Tensor([[29, 45], [36, 56]])")
        self.assertEqual((self.tensor @ self.tensor.permute(1, 0)).__repr__(), "Tensor([[34, 42], [42, 52]])")
        self.assertEqual((self.tensor2.reshape(2, 2) @ self.tensor2.reshape(2, 2)).__repr__(), "Tensor([[7, 10], [15, 22]])")
        self.assertEqual((self.tensor3 @ self.tensor3.permute(1, 0)).__repr__(), "Tensor([[17, 22, 27], [22, 29, 36], [27, 36, 45]])")
        self.assertEqual((self.tensor3.permute(1, 0) @ self.tensor3).__repr__(), "Tensor([[14, 32], [32, 77]])")

    def test_prod(self):
        self.assertEqual(self.tensor.prod(0).__repr__(), "Tensor([[12, 30]])")
        self.assertEqual(self.tensor.prod(1).__repr__(), "Tensor([[15], [24]])")
        self.assertEqual(self.tensor.prod(0, keepdim=False).__repr__(), "Tensor([12, 30])")
        self.assertEqual(self.tensor.prod(1, keepdim=False).__repr__(), "Tensor([15, 24])")

    def test_max(self):
        self.assertEqual(self.tensor.max(0)[0].__repr__(), "Tensor([[4, 6]])")
        self.assertEqual(self.tensor.max(0)[1].__repr__(), "Tensor([[1, 1]])")
        self.assertEqual(self.tensor.max(0, keepdim=False)[0].__repr__(), "Tensor([4, 6])")
        self.assertEqual(self.tensor.max(0, keepdim=False)[1].__repr__(), "Tensor([1, 1])")
        self.assertEqual(self.tensor2.max(0)[0].__repr__(), "Tensor([[[3, 4]]])")
        self.assertEqual(self.tensor2.max(0)[1].__repr__(), "Tensor([[[1, 1]]])")

if __name__ == '__main__':
    unittest.main()
