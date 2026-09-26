import numpy as np

def richardson(A, b, x0=None, tau=None, tol=1e-8, max_iter=10000, verbose=True):
    """
    Решение СЛАУ Ax = b методом установления (Ричардсона).

    Параметры
    ---------
    A        : (n,n) ndarray, матрица системы (желательно SPD)
    b        : (n,) ndarray, правая часть
    x0       : (n,) ndarray, начальное приближение (по умолчанию — нули)
    tau      : float, шаг итерации. Если None — берётся оптимальный по спектру.
    tol      : float, критерий остановки по невязке ||b - Ax||
    max_iter : int, максимум итераций
    verbose  : bool, печать прогресса

    Возвращает
    ----------
    x        : приближённое решение
    iters    : число выполненных итераций
    history  : список норм невязки по итерациям
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = b.size

    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.array(x0, dtype=float)

    # Автоматический выбор оптимального шага по спектру
    if tau is None:
        # Для SPD матрицы собственные значения вещественны и положительны
        eigvals = np.linalg.eigvalsh((A + A.T) / 2.0)
        lam_min = max(eigvals.min(), 1e-15)
        lam_max = eigvals.max()
        tau = 2.0 / (lam_min + lam_max)
        if verbose:
            print(f"[auto] λ_min={lam_min:.6g}, λ_max={lam_max:.6g}, "
                  f"τ_opt={tau:.6g}, κ={lam_max/lam_min:.3g}")

    history = []
    for k in range(1, max_iter + 1):
        r = b - A @ x                 # невязка
        rnorm = np.linalg.norm(r)
        history.append(rnorm)

        if rnorm < tol:
            if verbose:
                print(f"Сошлось за {k-1} итераций, ||r|| = {rnorm:.3e}")
            return x, k - 1, history

        x = x + tau * r               # x_{k+1} = x_k + τ(b - A x_k)

    if verbose:
        print(f"Достигнут предел итераций ({max_iter}), ||r|| = {history[-1]:.3e}")
    return x, max_iter, history


# ======================= ПРИМЕРЫ ДЛЯ ПРОВЕРКИ =======================
if __name__ == "__main__":

    # -------- Пример 1: маленькая SPD-система --------
    print("=" * 60)
    print("ПРИМЕР 1: симметричная положительно определённая 4x4")
    print("=" * 60)

    A1 = np.array([
        [ 4.0, -1.0,  0.0,  0.0],
        [-1.0,  4.0, -1.0,  0.0],
        [ 0.0, -1.0,  4.0, -1.0],
        [ 0.0,  0.0, -1.0,  3.0],
    ])
    b1 = np.array([1.0, 2.0, 3.0, 4.0])

    x_exact = np.linalg.solve(A1, b1)
    x_num, iters, hist = richardson(A1, b1, tol=1e-10)

    print("Точное решение      :", np.round(x_exact, 6))
    print("Метод Ричардсона    :", np.round(x_num, 6))
    print("Итераций            :", iters)
    print("Ошибка ||x - x*||   :", np.linalg.norm(x_num - x_exact))
    print()

    # -------- Пример 2: плохо обусловленная матрица (Гильберт) --------
    print("=" * 60)
    print("ПРИМЕР 2: плохо обусловленная матрица Гильберта 6x6")
    print("=" * 60)

    n = 6
    A2 = np.array([[1.0/(i + j + 1) for j in range(n)] for i in range(n)])
    b2 = A2 @ np.ones(n)   # точное решение — вектор из единиц

    x_num2, iters2, hist2 = richardson(A2, b2, tol=1e-6, max_iter=20000)

    print("Точное решение      :", np.ones(n))
    print("Метод Ричардсона    :", np.round(x_num2, 6))
    print("Итераций            :", iters2)
    print("Ошибка ||x - x*||   :", np.linalg.norm(x_num2 - np.ones(n)))

    # -------- Пример 3: сравнение с numpy --------
    print()
    print("=" * 60)
    print("ПРИМЕР 3: сверка с numpy.linalg.solve")
    print("=" * 60)

    rng = np.random.default_rng(0)
    M = rng.normal(size=(10, 10))
    A3 = M.T @ M + 10 * np.eye(10)      # SPD, хорошо обусловлена
    b3 = rng.normal(size=10)

    x_np   = np.linalg.solve(A3, b3)
    x_rich, it3, _ = richardson(A3, b3, tol=1e-10)

    print("||x_rich - x_numpy|| =", np.linalg.norm(x_rich - x_np))
    print("Итераций            :", it3)
