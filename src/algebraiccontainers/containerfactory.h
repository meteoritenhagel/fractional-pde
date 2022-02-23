#ifndef FILE_COLMATRIXFACTORY
#define FILE_COLMATRIXFACTORY

#include "algebraiccontainers.h"
#include "coefficientmatrix.h"
#include "../processingunit/processingunit.h"
#include "../devicedata/devicedata.h"

template<class floating>
class AlgebraicMatrix;

template<class floating>
class AlgebraicVector;

/**
 * Class ContainerFactory is a helper class used for easy creation of the container types
 * declared in algebraiccontainers.h. Only the processingUnit has to given to this class,
 * and each "create" function then builds the container with the right data type and
 * right memory location.
 *
 * All containers are returned by std::unique_ptr, except when an instance of AlgebraicVector is
 * created using create_array(...). In this case, a std::shared_ptr is returned.
 *
 * @tparam floating Floating point type
 */
template<class floating>
class ContainerFactory {
public:
    using ColMatrixPointer = std::unique_ptr<AlgebraicMatrix<floating>>;
    using ColMatrixColumnPointer = std::shared_ptr<AlgebraicVector<floating>>;
    using CoefficientMatrixPointer = std::unique_ptr<CoefficientMatrix<floating>>;
    using SizeType = typename DeviceDataDevice<floating>::SizeType;

    explicit ContainerFactory(const ProcessingUnit<floating> processing_unit);
    ~ContainerFactory() = default;

    /** Default factory
     * @return returns a default AlgebraicMatrix with (0, 0) elements.
     */
    ColMatrixPointer create_matrix() const;

    /** Factory, initializes elements with value @p val
     *  @param[in] num_rows  number of columns
     *  @param[in] num_cols  number of columns
     *  @param[in] val   initial value for all matrix elements
     */
    ColMatrixPointer create_matrix(const SizeType num_rows, const SizeType num_cols, const floating val = 0) const;

    /**  Factory, creates an AlgebraicMatrix from vector @p vec.
     *   The size of @p vec must be a multiple of @p num_rows.
     * @param[in] vec vector
     */
    ColMatrixPointer create_matrix(const SizeType num_rows, std::vector<floating> const &vec) const;

    /**  Factory, creates an individual column / AlgebraicVector with @p size elements,
     * which are initialized with the value @p val.
     * @param size number of elements
     * @param val initial value vor all individual elements
     */
    ColMatrixColumnPointer create_array(const SizeType size, const floating val = 0) const;

    /**  Factory, creates an individual column / AlgebraicVector for a fractional PDE
     * with @p N time intervals, which are initialized with the value @p val.
     * @param N number of time intervals in the equidistant time grid
     * @param alpha anomalous diffusion coefficient
     */
    CoefficientMatrixPointer create_coefficient_matrix(const SizeType N, const floating alpha) const;

    /**  Returns the current instance's processing unit.
     * @return current processing unit
     */
    ProcessingUnit<floating> get_processing_unit() const;

private:
    ProcessingUnit<floating> _processing_unit; //!< Target processing unit defining memory allocation location and linear algebra calls for the containers
};

#include "containerfactory.hpp"
#endif
