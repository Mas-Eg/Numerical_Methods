import numpy as np

np.set_printoptions(precision=4, suppress=True)


def gauss_solve(A, b, partial_pivoting=True, tol=0.0):
    """
    Решение СЛАУ Ax = b методом Гаусса.

    partial_pivoting=True  — с частичным выбором ведущего элемента по столбцу.
    partial_pivoting=False — naive-Гаусс без выбора ведущего элемента.
    tol — порог для обнаружения нулевого ведущего элемента.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).copy()

    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Матрица A должна быть квадратной")
    if b.size != n:
        raise ValueError("Размер b не совпадает с размером A")

    for k in range(n):
        if partial_pivoting:
            p = k + np.argmax(np.abs(A[k:, k]))
            if abs(A[p, k]) <= tol:
                raise np.linalg.LinAlgError("Матрица вырождена или почти вырождена")
            if p != k:
                A[[k, p], :] = A[[p, k], :]
                b[[k, p]] = b[[p, k]]
        else:
            if abs(A[k, k]) <= tol:
                raise np.linalg.LinAlgError("Нулевой ведущий элемент")

        # Прямой ход: исключаем неизвестные ниже диагонали
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # Обратный ход
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(A[i, i]) <= tol:
            raise np.linalg.LinAlgError("Матрица вырождена")
        x[i] = (b[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]

    return x


def rel_error(x, x_true):
    return np.linalg.norm(x - x_true) / np.linalg.norm(x_true)


def rel_residual(A, x, b):
    return np.linalg.norm(A @ x - b) / np.linalg.norm(b)

print("=== 1. Хорошо обусловленная система ===")

A = np.array([[4., 1., 2.],
              [1., 3., 0.],
              [2., 0., 5.]])

x_true = np.array([1., 2., 3.])
b = A @ x_true

x = gauss_solve(A, b)

print("cond(A) =", np.linalg.cond(A))
print("x       =", x)
print("ошибка  =", rel_error(x, x_true))
print("невязка =", rel_residual(A, x, b))

print("\n=== 2. Маленький ведущий элемент ===")

A = np.array([[1e-20, 1.],
              [1.,     1.]])
b = np.array([1., 2.])

x_no_pivot = gauss_solve(A, b, partial_pivoting=False)
x_pivot    = gauss_solve(A, b, partial_pivoting=True)
x_np       = np.linalg.solve(A, b)

print("cond(A) =", np.linalg.cond(A))
print("без выбора          :", x_no_pivot)
print("с частичным выбором :", x_pivot)
print("numpy.linalg.solve  :", x_np)

print("\n=== 3. Плохо обусловленная матрица Гильберта ===")

n = 12
A = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])

x_true = np.ones(n)
b = A @ x_true

x_gauss = gauss_solve(A, b)
x_np = np.linalg.solve(A, b)

print("n =", n)
print("cond(A) =", np.linalg.cond(A))
print("ошибка  (Gauss) =", rel_error(x_gauss, x_true))
print("невязка (Gauss) =", rel_residual(A, x_gauss, b))
print("ошибка  (numpy) =", rel_error(x_np, x_true))
print("невязка (numpy) =", rel_residual(A, x_np, b))
print("x[:5] =", x_gauss[:5])