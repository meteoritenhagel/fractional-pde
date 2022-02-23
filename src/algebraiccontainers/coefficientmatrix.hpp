#include "../fractional_pde.h"

#include <cmath>

template<class floating>
CoefficientMatrix<floating> operator*(const floating scalar, const CoefficientMatrix<floating> &B)
{
    CoefficientMatrix<floating> tmp(B);
    tmp.scale(scalar);

    return tmp;
}

// public:

template<class floating>
CoefficientMatrix<floating>::CoefficientMatrix(const ProcessingUnit<floating> processing_unit, const SizeType size, const floating alpha)
: _D(initialize_D(processing_unit, size, alpha))
{}

template<class floating>
typename CoefficientMatrix<floating>::SizeType CoefficientMatrix<floating>::size() const
{
    return _D.size();
}

template<class floating>
ProcessingUnit<floating> CoefficientMatrix<floating>::get_processing_unit() const
{
    return _D.get_processing_unit();
}

template<class floating>
ContainerFactory<floating> CoefficientMatrix<floating>::get_container_factory() const
{
    return _D.get_container_factory();
}

template<class floating>
AlgebraicMatrix<floating> CoefficientMatrix<floating>::get_dense_representation() const
{
    auto denseMatrix = *get_container_factory().create_matrix(size(), size());

    for (SizeType i = size()-2; i >= 2; --i)
    {
        for (SizeType j = 0; j < i + 2; ++j)
        {
            denseMatrix(i,j) = _D[size()-1 - i + j - 1];
        }
    }

    return denseMatrix;

}

template<class floating>
floating CoefficientMatrix<floating>::operator()(const SizeType i, const SizeType j) const
{
    floating entry = 0;

    if (i != 0 && i != 1 && i != size())
        if (j < i + 2)
            entry = _D[size()-1 - i + j - 1];

    return entry;
}

template<class floating>
void CoefficientMatrix<floating>::scale(const floating scalar)
{
    _D.scale(scalar);
    return;
}

template<class floating>
AlgebraicVector<floating> CoefficientMatrix<floating>::operator*(const AlgebraicVector<floating> &rhs)
{
    assert(this->size() == rhs.size() && "ERROR: Dimension mismatch. Cannot perform multiplication.");
    const auto N = size()-3;
    auto result = *get_container_factory().create_column(size());

    for (SizeType i = N+1; i >= 2; --i)
    {
        floating temp = 0;
        for (SizeType j = 0; j < i+2; ++j)
        {
            temp += _D[size()-1 - i + j -1] * rhs[j];
        }
        result[i] = temp;
    }

    return result;
}

// private:

template<class floating>
AlgebraicVector<floating> CoefficientMatrix<floating>::initialize_D(const ProcessingUnit<floating> processing_unit, const SizeType size, const floating alpha)
{
    ContainerFactory<floating> factory(processing_unit);
    auto D = *factory.create_array(size+3);

    for (unsigned j = 0; j < size + 2; j++)
    {
        D[j] = CoefficientMatrix<floating>::coef_d(size + 2 - j, size, alpha);
    }
    D[size + 2] = -1;
    return D;
}

template<class floating>
floating CoefficientMatrix<floating>::coef_a(const int m, const floating alpha)
{
    return pow(m + 1, 3 - alpha) - pow(m, 3 - alpha);
}

template<class floating>
floating CoefficientMatrix<floating>::coef_b(const int m, const floating alpha)
{
    return (3 - alpha) * pow(m, static_cast<double>(2 - alpha));
}
template<class floating>
floating CoefficientMatrix<floating>::coef_b_prime(const int m, const floating alpha)
{
    return 0.5 * (3 - alpha) * (2 - alpha) * pow(m, 1 - alpha);
}

template<class floating>
floating CoefficientMatrix<floating>::coef_d(const int m, const int k, const floating alpha)
{
    floating rv(-1234567);
    if (m == 1)
        rv = 3 * coef_a(0, alpha) - (coef_a(1, alpha) + 2 * coef_b(0, alpha));
    else if (m == 2)
        rv = 3 * coef_a(1, alpha) + coef_b(0, alpha) - (3 * coef_a(0, alpha) + coef_a(2, alpha) + coef_b_prime(0, alpha));
    else if (m >= 3 && m <= k - 1)
        rv = 3 * coef_a(m - 1, alpha) + coef_a(m - 3, alpha) - (3 * coef_a(m - 2, alpha) + coef_a(m, alpha));
    else if (m == k)
        rv = 3 * coef_a(k - 1, alpha) + coef_a(k - 3, alpha) - (3 * coef_a(k - 2, alpha) + coef_b(k, alpha) +
                coef_b_prime(k, alpha));
    else if (m == k + 1)
        rv = coef_a(k - 2, alpha) + 2 * coef_b(k, alpha) - 3 * coef_a(k - 1, alpha);
    else if (m == k + 2)
        rv = coef_a(k - 1, alpha) + coef_b_prime(k, alpha) - coef_b(k, alpha);

    return rv;
}

