#ifndef COLMATRIX_COEFFICIENTMATRIX_H_
#define COLMATRIX_COEFFICIENTMATRIX_H_

#include "algebraiccontainers.h"
#include "../processingunit/processingunit.h"

template<class floating>
class CoefficientMatrix;

template<class floating>
class AlgebraicMatrix;

template<class floating>
class AlgebraicVector;

template<class floating>
class ContainerFactory;

/**
 * Calculates the product of a scalar @p scalar and a CoefficientMatrix @p B.
 * The result is stored in a newly allocated object.
 *
 * @tparam floating Floating point type
 * @param scalar scalar
 * @param B CoefficientMatrix
 * @return scalar * B allocated as a new object
 */
template<class floating>
CoefficientMatrix<floating> operator*(const floating scalar, const CoefficientMatrix<floating> &B);

/**
 * The class CoefficientMatrix is a matrix of special form, which emerges when discretizing Caputo fractional
 * differential equations using FEM. They have the following form:
 *
 *       [ 0       ...                      0]
 *       [ 0       ...                      0]
 *       [ d_3     d_2  d_1   -1   0  ...   0]
 *  D =  [ d_4     d_3  d_2  d_1  -1  0 ... 0]
 *       [ ...                          ... 0]
 *       [ d_(N+2) ...                  ...-1]
 *       [ 0       ...                  ... 0]
 *
 * The values d_1 ... d_(N+2) are calculated during initialization
 * and only depends on the number of equidistant time steps N and
 * the anomalous diffusion exponent alpha.
 *
 * @tparam floating Floating point type
 */
template<class floating>
class CoefficientMatrix {
public:
    using SizeType = typename AlgebraicMatrix<floating>::SizeType;

    /**
     * Constructor.
     * @param processing_unit Target processing_unit
     * @param size Number of equidistant time steps N
     * @param alpha anomalous diffusion exponent
     */
    CoefficientMatrix(const ProcessingUnit<floating> processing_unit, const SizeType size, const floating alpha);

    /**
     * Destructor
     */
    ~CoefficientMatrix() = default;

    /**
     * Return number of time intervals in equidistant time grid.
     * @return number of equidistant time intervals
     */
    SizeType size() const;

    /**
     * Return this instance's processing unit.
     * @return current processing unit
     */
    ProcessingUnit<floating> get_processing_unit() const;

    /**
     * Return this instance's container factory.
     * @return current container factory
     */
    ContainerFactory<floating> get_container_factory() const;

    /**
     * Copies the abstract matrix type to a dense AlgebraicMatrix.
     * @return dense matrix containing the matrix's values
     */
    AlgebraicMatrix<floating> get_dense_representation() const;

    /**
     * Access element in row @param i and column @param j of the current abstract matrix.
     * @param i row index
     * @param j column index
     * @return A_ij
     */
    floating operator()(const SizeType i, const SizeType j) const;

    /** Scales matrix with scalar @p scalar.
    * @param[in] scalar scalar factor
    */
    void scale(const floating scalar);

    /** Multiplies this matrix with an AlgebraicVector @p rhs.
    * @param[in] B AlgebraicVector
    * @return this * rhs in a newly allocated AlgebraicVector.
    */
    AlgebraicVector<floating> operator*(const AlgebraicVector<floating> &rhs);

private:
    AlgebraicVector<floating> _D; //!< The data values d_1, ..., d_(N+2) defining the abstract matrix

    /**
     * Initialize data member with the correct values.
     * @param processing_unit target processing unit
     * @param size number of equidistant time steps
     * @param alpha anomalous diffusion coefficient
     * @return the correct AlgebraicVector for initialization
     */
    static AlgebraicVector<floating> initialize_D(const ProcessingUnit<floating> processing_unit,
                                                  const SizeType size, const floating alpha);

    /** Helper function for calculation of data member.
     *
     * @param m integer index
     * @param alpha anomalous diffusion coefficient
     * @return (m + 1)^(3 - alpha) - m^(3 - alpha);
     */
    static floating coef_a(const int m, const floating alpha);

    /** Helper function for calculation of data member.
     *
     * @param m integer index
     * @param alpha anomalous diffusion coefficient
     * @return (3 - alpha) * m^(2 - alpha);
     */
    static floating coef_b(const int m, const floating alpha);

    /** Helper function for calculation of data member.
     *
     * @param m integer index
     * @param alpha anomalous diffusion coefficient
     * @return 0.5 * (3 - alpha) * (2- alpha) * m^(1 - alpha);
     */
    static floating coef_b_prime(const int m, const floating alpha);

    /** Helper function for calculation of data member.
     * Chooses the right coef_* function according to @p m and @p k.
     *
     * @param m integer index
     * @param k maximum index (in this context, normally the number of equidistant time steps)
     * @param alpha anomalous diffusion coefficient
     * @return the correct coef_* function according to @p m and @p k
     */
    static floating coef_d(const int m, const int k, const floating alpha);
};

#include "coefficientmatrix.hpp"

#endif /* COLMATRIX_COEFFICIENTMATRIX_H_ */
