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
 * created using createColumn(...). In this case, a std::shared_ptr is returned.
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

    explicit ContainerFactory(const ProcessingUnit<floating> processingUnit);
    ~ContainerFactory() = default;

    /** Default factory
     * @return returns a default AlgebraicMatrix with (0, 0) elements.
     */
    ColMatrixPointer createMatrix() const;

    /** Factory, initializes elements with value @p val
     *  @param[in] nrow  number of columns
     *  @param[in] ncol  number of columns
     *  @param[in] val   initial value for all matrix elements
     */
    ColMatrixPointer createMatrix(const SizeType nrow, const SizeType ncol, const floating val = 0) const;

    /**  Factory, creates an AlgebraicMatrix from vector @p u.
     * @param[in] u  vector
     */
    ColMatrixPointer createMatrix(const SizeType nrow, std::vector<floating> const &u) const;

    /**  Factory, creates an individual column / AlgebraicVector with @param size elements,
     * which are initialized with the value @param val.
     * @param size number of elements
     * @param val initial value vor all individual elements
     */
    ColMatrixColumnPointer createColumn(const SizeType size, const floating val = 0) const;

    /**  Factory, creates an individual column / AlgebraicVector with @param size elements,
     * which are initialized with the value @param val.
     * @param size number of elements
     * @param val initial value vor all individual elements
     */
    CoefficientMatrixPointer createCoefficientMatrix(const SizeType size, const floating alpha) const;

    /**  Returns the current instance's processing unit.
     * @return current processing unit
     */
    ProcessingUnit<floating> getProcessingUnit() const;

private:
    ProcessingUnit<floating> _processingUnit; //!< Target processing unit defining memory allocation location and linear algebra calls for the containers
};

#include "containerfactory.hpp"
#endif
