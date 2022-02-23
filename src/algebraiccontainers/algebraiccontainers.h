#ifndef FILE_COLMATRIX
#define FILE_COLMATRIX

#include "../processingunit/processingunit.h"
#include "../devicedata/devicedata.h"

#include <memory>
#include <vector>
#include "containerfactory.h"

// forward declaration
template<class floating>
class AlgebraicMatrix;

// forward declaration
template<class floating>
class AlgebraicVector;

/**
 * Use BlockVector as an alias for AlgebraicMatrix,
 * since the occurring long vectors, consisting of blocks,
 * can be handled easier when interpreted as a matrix.
 */
template<class T>
using BlockVector = AlgebraicMatrix<T>;

/**
 * Calculates the product of a scalar @p alpha and an AlgebraicMatrix @p B.
 * The result is stored in a newly allocated object.
 *
 * @tparam floating Floating point type
 * @param alpha Scalar
 * @param B AlgebraicMatrix
 * @return alpha * B
 */
template<class floating>
AlgebraicMatrix<floating> operator*(const floating alpha, const AlgebraicMatrix<floating> &B);

/**
 * Calculates the scalar (or "inner") product <A, B>
 * of BlockVectors @p A and @p B.
 *
 * @tparam floating Floating point type
 * @param A first vector (or matrix)
 * @param B second vector (or matrix)
 * @return the scalar product <A, B>
 */
template<class floating>
floating scalarProduct(const BlockVector<floating> &A, const BlockVector<floating> &B);

/**
 * Calculates the scalar (or "inner") product <A, B>
 * of AlgebraicVectors @p A and @p B.
 *
 * @tparam floating Floating point type
 * @param A first vector
 * @param B second vector
 * @return the scalar product <A, B>
 */
template<class floating>
floating scalarProduct(const AlgebraicVector<floating> &A, const AlgebraicVector<floating> &B);

/**
 * The class AlgebraicMatrix provides an easy-to-use column-wise stored matrix type.
 * An instance of ProcessingUnit is passed to determine where the matrix operations
 * are calculated (i.e. for determining execution on Cpu or Gpu).
 *
 * If compile variable PLU is set, the inverse is never calculated, but PLU factorization is used
 * for solving the according system of equations.
 * Otherwise, the matrix inverse is calculated once and then directly used for calculations.
 *
 * @tparam floating Floating point type
 */
template<class floating>
class AlgebraicMatrix {
public:
    using SizeType = typename DeviceDataDevice<floating>::SizeType;
    using MatrixDataType = DeviceMatrix<floating>;
    using ArrayOfColumns = std::vector<AlgebraicVector<floating>>;

    /** Default constructor, constructs element by passing the vector of arrays.
     *
     *  @param[in] processing_unit  processing_unit (i.e. for determining Cpu or Gpu execution)
     * 	@param[in] A                data of matrix to construct
     */
    AlgebraicMatrix(const ProcessingUnit<floating>& processing_unit, const MatrixDataType& A);


    /**
     * Copy constructor
     */
    AlgebraicMatrix(const AlgebraicMatrix &other);

    /**
     * Move constructor
     */
    AlgebraicMatrix(AlgebraicMatrix &&) = default;

    /**
     * Destructor
     */
    virtual ~AlgebraicMatrix() = default;

    /**
     * This function moves the members of the current instance
     * to the data location associated with @param processing_unit.
     * @param processing_unit Target ProcessingUnit
     */
    void move_to(const ProcessingUnit<floating> processing_unit);

    /**
     * Copy assignment operator.
     */
    AlgebraicMatrix& operator=(const AlgebraicMatrix &other);

    /**
     * Move assignment operator.
     */
    AlgebraicMatrix& operator=(AlgebraicMatrix&&) = default;

    /**
     * Resizes member _A to be of dimensions (@param num_rows, @param num_cols)
     * @param num_rows new number of rows
     * @param num_cols new number of columns
     * @return reference to current instance
     */
    AlgebraicMatrix<floating>& resize(const SizeType num_rows, const SizeType num_cols);

    /** Returns pointer to 0th element of 0th column if it exists.
     *
     * @return pointer to begin of vector data.
     */
    floating* data();

    /** Returns pointer to 0th element of 0th column if it exists.
     *
     * @return pointer to begin of vector data.
     */
    const floating* data() const;

    /** Access to column @p col of this matrix.
     *  This allows a 2D access [col][row] to the matrix.
     *
     * @param[in] col column index
     *
     * @return vector containing col-th column.
     */
    AlgebraicVector<floating>& operator[](const SizeType col);

    /** Access to const column @p col of this matrix.
     *  This allows a 2D access [col][row] to the matrix.
     *
     * @param[in] col column index
     *
     * @return vector containing col-th column.
     */
    AlgebraicVector<floating> const & operator[](const SizeType col) const;

    /** Access to element in row @param i and column @param j.
     * This is equivalent to a 2D access [j][i] to the matrix.
     *
     * @param[in] i row index
     * @param[in] j column index
     *
     * @return element in row i and column j
     */
    floating& operator()(const SizeType i, const SizeType j);

    /** Access to const element in row @p i and column @p j.
     * This is equivalent to a 2D access [j][i] to the matrix.
     *
     * @warning the matrix must already be in RAM before calling this function
     *
     * @param[in] i row index
     * @param[in] j column index
     *
     * @return element in row i and column j
     */
    floating const & operator()(const SizeType i, const SizeType j) const;

    /**
     * Copies the entries of matrix row @p i to a new AlgebraicVector.
     * @warning the matrix must already be in RAM before calling this function
     * @param i row index
     * @return AlgebraicVector containing the @p i-th row of the matrix
     */
    AlgebraicVector<floating> getRow(const SizeType i) const;

    /** Return the elements of the matrix in form of a long std::vector<floating>,
     *  containing column after column.
     *
     * @return vector containing matrix elements.
     */
    std::vector<floating> values() const;

    /**
     * Returns a copy of the matrix, which is resized to be a long vector,
     * containing column after column.
     *
     * @return AlgebraicMatrix containing matrix elements like a long vector.
     */
    AlgebraicMatrix<floating> flat() const;

    /**
     * Return the number of rows of the current instance.
     * @return number of rows
     */
    SizeType get_num_rows() const;

    /**
     * Return the number of columns of the current instance.
     * @return number of columns
     */
    SizeType get_num_cols() const;

    /** Get number of all elements contained in the AlgebraicMatrix.
     *
     * @return number of elements.
     */
    SizeType get_num_elements() const;

    /** Checks whether the current matrix is a square matrix.
     * @return true if the matrix is square.
     */
    bool is_square() const;

    /**
     * Returns the container factory which can be used to create new containers
     * on the same device with the same processing unit.
     * @return container factory
     */
    ContainerFactory<floating> get_container_factory() const;

    /**
    * Returns the current instance's processing unit.
    * @return processing unit
    */
    ProcessingUnit<floating> get_processing_unit() const;

    /** Returns const reference to the inverse, which is calculated, if needed.
     * @return const reference to the inverse
     */
    AlgebraicMatrix const & get_inverse() const;

    /**
     * Calculates the Euclidean norm of the current matrix's data
     * interpreted as a vector.
     * @return Euclidean norm
     */
    floating get_euclidean_norm() const;

    /**
     * Calculates the element with maximum absolute value.
     * @return element with maximum absolute value
     */
    floating get_maximum_norm() const;

    /**
    * Returns the contents in a human-readable format as a string.
    * Display is roughly the following: "@param name = (<contents>)"
    * @param name Name to display
    * @return string representation of current instance
    */
    std::string display(std::string name) const;

    /**
     * Adds scalar times matrix @p B to this matrix.
     * @param[in] B matrix
     * @param[in] scalar = 1.0 by default, scalar
     * @return A = A + scalar * B.
     */
    AlgebraicMatrix<floating>& add(const AlgebraicMatrix &B, const floating scalar = 1.0);

    /**
     * Adds matrix @p B to this matrix from column @p col_begin to inclusively @p col_end-1.
     * @param[in] col_begin column start
     * @param[in] col_end   column end
     * @param[in] B         matrix to add
     * @warning the number of rows must be equal
     */
    void updateAdd(const SizeType col_begin, const SizeType col_end, const AlgebraicMatrix &B);

    /**
     * Adds this matrix to matrix @p B and returns the result.
     * @param[in] B matrix
     * @return A + B in a newly allocated AlgebraicMatrix
     */
    AlgebraicMatrix operator+(const AlgebraicMatrix &B) const;

    /**
     * Adds matrix @p B to this matrix.
     * @param[in] B matrix
     * @return A = A + B.
     */
    AlgebraicMatrix& operator+=(const AlgebraicMatrix &B);

    /**
     * Subtracts matrix @p B from this matrix.
     * @param[in] B matrix
     * @return A - B in a newly allocated AlgebraicMatrix
     */
    AlgebraicMatrix operator-(const AlgebraicMatrix &B) const;

    /**
     * cales matrix with scalar @p alpha.
     * @param[in] alpha scalar factor
     */
    void scale(const floating alpha);

    /**
     * Multiplies this matrix with matrix @p B.
     * @param[in] B matrix
     * @return the matrix product A * B in a newly allocated AlgebraicMatrix.
     */
    AlgebraicMatrix<floating> mult(const AlgebraicMatrix &B) const;

    /**
     * Multiplies this matrix with vector @p vec.
     * @param[in] vec vector
     * @return A * vec as newly allocated std::vector.
     */
    AlgebraicVector<floating> mult(const AlgebraicVector<floating> &vec) const;

    /**
     * Multiplies this matrix with AlgebraicMatrix @p B.
     * @param[in] B AlgebraicMatrix
     * @return A * B in a newly allocated AlgebraicMatrix.
     */
    AlgebraicMatrix operator*(const AlgebraicMatrix &B) const;

    /**
     * Multiplies this matrix with vector @p vec.
     * @param[in] vec vector
     * @return A * vec in a newly allocated AlgebraicVector.
     */
    AlgebraicVector<floating> operator*(const AlgebraicVector<floating> &vec) const;

    /** Multiplies the inverse of this square matrix with matrix @p B.
     *  This operator is the equivalent to the operator\ in Matlab.
     *  A has to be square.
     *
     *  The inverse of this matrix is never calculated directly,
     *  if compile variable PLU is set. Then, a system of equations is solved with rhs @p B.
     *  Otherwise, the inverse is calculated and then cached internally.
     *
     * @param[in] B matrix
     *
     * @return A\\B = A^(-1) * B in a newly allocated AlgebraicMatrix.
     */
    AlgebraicMatrix operator/(const AlgebraicMatrix<floating> &B) const;

    /**
     * Multiplies the inverse of this square matrix with matrix @p B.
     *  This operator is the equivalent to the operator\ in Matlab.
     *  A has to be square.
     *
     *  The inverse of this matrix is never calculated directly,
     *  if compile variable PLU is set. Then, a system of equations is solved with rhs @p B.
     *  Otherwise, the inverse is calculated and then cached internally.
     *
     *  B is overwritten with the result A\\B = A^(-1) * B.
     *
     * @param[in,out] B matrix
     */
    void invTimes(AlgebraicMatrix<floating> &B) const;

    /**
     * Multiplies the inverse of this square matrix with std::vector @p vec.
     *  This operator is the equivalent to the operator\ in Matlab.
     *  A has to be square.
     *
     *  The inverse of this matrix is never calculated directly,
     *  if compile variable PLU is set. Then, a system of equations is solved with rhs @p vec.
     *  Otherwise, the inverse is calculated and then cached internally.
     *
     *  The return type is AlgebraicMatrix instead of AlgebraicVector,
     *  since internally the std::vector is then converted to an
     *  AlgebraicMatrix.
     *
     * @param[in] vec std::vector
     *
     * @return A\\vec = A^(-1) * vec in a newly allocated AlgebraicMatrix.
     *
     */
    AlgebraicMatrix operator/(const std::vector<floating>& vec) const;

    /**
     * Multiplies the inverse of this square matrix with AlgebraicVector @p vec.
     *  This operator is the equivalent to the operator\ in Matlab.
     *  A has to be square.
     *
     *  The inverse of this matrix is never calculated directly,
     *  if compile variable PLU is set. Then, a system of equations is solved with rhs @p vec.
     *  Otherwise, the inverse is calculated and then cached internally.
     *
     * @param[in] vec AlgebraicVector
     *
     * @return A\\vec = A^(-1) * vec in a newly allocated AlgebraicVector.
     */
    AlgebraicVector<floating> operator/(const AlgebraicVector<floating> &vec) const;

    /**
     * Multiplies the inverse of this square matrix with AlgebraicVector @p vec.
     *  This operator is the equivalent to the operator\ in Matlab.
     *  A has to be square.
     *
     *  The inverse of this matrix is never calculated directly,
     *  if compile variable PLU is set. Then, a system of equations is solved with rhs @p vec.
     *  Otherwise, the inverse is calculated and then cached internally.
     *
     * @param[in] B vector
     * @param[out] result is overwritten with the result A\\vec = A^(-1) * vec.
     */
    void inverse_times(const AlgebraicVector<floating> &vec, AlgebraicVector<floating> &result) const;

    /**
     * Accesses the inverse (which is calculated, if needed) by reference.
     */
    const AlgebraicMatrix& access_inverse() const;

private:
    ContainerFactory<floating> _container_factory; //!< Container factory used for the creation of new matrices with the right memory location and processing unit.
    MatrixDataType _A;    //!< matrix values
    mutable std::unique_ptr<AlgebraicMatrix<floating>> _inverse; //!< If compile variable PLU is set, this is the factorized matrix for inversion. Otherwise this contains the actual inverse. Only stored and computed if needed.

#ifdef PLU //TODO: CHECK IF THIS IS REALLY NECESSARY
public:
#endif
    mutable DeviceArray<int> _permutation; //!< permutation vector from matrix factorization.

    ArrayOfColumns _array_of_columns;

    ArrayOfColumns initialize_array_of_columns() const;
    void reset_array_of_columns();

    /**
     *    Accesses the inverse, which is calculated, if needed, passed by reference and hence modifyable.
     *
     *    @warning This method should only be used to modify the inverse such that it remains the inverse of
     *    @warning the current AlgebraicMatrix! Otherwise, unexpected behaviour when using operator/ will occur.
     *    @warning For getting the inverse, use get_inverse() instead.
     */
    AlgebraicMatrix& access_inverse();

    /**
     * Returns whether the inverse matrix is valid
     * @return true if the inverse matrix is valid
     */
    bool is_inverse_set() const;

    /**
     * Calculates the inverse (resp. the PLU factorization)
     */
    void recalculate_inverse() const;

    /**   Resets the inverse.
     *
     *    @warning This method should be used in any method changing the matrix
     *    if the inverse gets invalid and is not currently needed.
     */
    void reset_inverse() const;
};

/**
 * Calculates the product of a scalar @p scalar and an AlgebraicVector @p B.
 *
 * @tparam floating Floating point type
 * @param scalar Scalar
 * @param B AlgebraicMatrix
 * @return scalar * B stored in a newly allocated object
 */
template<class floating>
AlgebraicVector<floating> operator* (floating const scalar, const AlgebraicVector<floating> &B);

/**
 * The class AlgebraicVector provides an easy-to-use vector type.
 * An instance of ProcessingUnit is passed to determine where the linear algebra operations
 * are calculated (i.e. for determining execution on Cpu or Gpu).
 *
 * @tparam floating Floating point type
 */
template<class floating>
class AlgebraicVector {
public:
    using SizeType = typename DeviceDataDevice<floating>::SizeType;
    using ArrayDataType = DeviceArray<floating>;
    using ArrayPointerType = std::shared_ptr<ArrayDataType>;

    /**
     * Constructor
     * @param processing_unit processing unit
     * @param data pointer to vector data
     */
    explicit AlgebraicVector(const ProcessingUnit<floating> processing_unit = nullptr, ArrayPointerType data = nullptr);

    /**
     * Copy constructor
     */
    AlgebraicVector(const AlgebraicVector &other);

    /**
     * Move constructor
     */
    AlgebraicVector(AlgebraicVector &&other) = default;

    /**
     * Destructor
     */
    virtual ~AlgebraicVector() = default;

    /**
     * This function moves the members of the current instance
     * to the data location associated with @param processing_unit.
     * @param processing_unit Target ProcessingUnit
     */
    AlgebraicVector<floating>& move_to(const ProcessingUnit<floating> processing_unit);

    /**
     * Copy assignment operator.
     */
    AlgebraicVector& operator=(const AlgebraicVector &other);

    /**
     * Move assignment operator
     */
    AlgebraicVector& operator=(AlgebraicVector &&other);

    /**
     * Returns pointer to 0th element of 0th column.
     * @return pointer to begin of vector data.
     */
    floating* data();

    /**
     * Returns const pointer to 0th element of 0th.
     * @return const pointer to begin of vector data.
     */
    const floating* data() const;

    /**
     * Access to element with index @p index of this vector.
     * @param[in] index element index
     * @return @p index-th element.
     */
    floating& operator[](const SizeType index);

    /**
     * Access to const element with index @p index of this vector.
     * @param[in] index element index
     * @return @p index-th element.
     */
    floating const & operator[](const SizeType index) const;

    /**
     * Get number of elements contained in the AlgebraicMatrix.
     * @return number of elements.
     */
    SizeType size() const;

    /**
     * Returns the container factory which can be used to create new containers
     * on the same device with the same processing unit.
     * @return container factory
     */
    ContainerFactory<floating> get_container_factory() const;

    /**
    * Returns the current instance's processing unit.
    * @return processing unit
    */
    ProcessingUnit<floating> get_processing_unit() const;

    /**
    * Calculates the Euclidean norm of the current vector.
    * @return Euclidean norm
    */
    floating get_euclidean_norm() const;

    /**
     * Calculates the element with maximum absolute value.
     * @return element with maximum absolute value
     */
    floating get_maximum_norm() const;

    /**
    * Returns the contents in a human-readable format as a string.
    *
    * Display is roughly the following: "@param name = (<contents>)"
    * @param name Name to display
    *
    * @return string representation of current instance
    */
    std::string display(std::string name) const;

    /**
     * Adds scalar times vector @p B to this matrix.
     * @param[in] B AlgebraicVector
     * @param[in] scalar = 1.0 by default, scalar
     * @return A = A + scalar * B.
     */
    AlgebraicVector& add(const AlgebraicVector &B, const floating scalar = 1.0);

    /**
     * Adds this vector to vector @p B and returns the result.
     * @param[in] B vector
     * @return A + B in a newly allocated AlgebraicVector
     */
    AlgebraicVector operator+(const AlgebraicVector &B) const;

    /**
     * Adds vector @p B to this vector.
     * @param[in] B vector
     * @return A = A + B.
     */
    AlgebraicVector& operator+=(const AlgebraicVector &B);

    /**
     * Subtracts vector @p B from this vector.
     * @param[in] B vector
     * @return A - B in a newly allocated AlgebraicVector
     */
    AlgebraicVector operator-(const AlgebraicVector &B) const;

    /**
     * Calculates the transpose of this vector times the matrix @p A.
     * @param[in] A matrix
     * @return (*this)^T * A in a newly allocated AlgebraicVector
     */
    AlgebraicVector operator*(const AlgebraicMatrix<floating> &A) const;

    /**
     * Scales current vector with scalar @p scalar.
     * @param[in] scalar scalar factor
     */
    void scale(const floating scalar);

private:
    ContainerFactory<floating> _container_factory; //!< container factory used for creating new containers with the same ProcessingUnit and memory location
    ArrayPointerType _data;    //!< vector values

    /**
     * Checks whether the vector defined by the member variable is valid
     * @return true if the vector are valid
     */
    bool is_valid() const;

    /**
     * Returns the reference to the array contained in the member data type.
     * @return reference to array
     */
    ArrayDataType& access_array();

    /**
     * Returns the const reference to the array contained in the member data type.
     * @return const reference to array
     */
    ArrayDataType const & access_array() const;

    /**
     * Initialize the current vector member data type by copying it from another instance @param other
     * @param other other vector
     * @return ArrayPointerType for initialization
     */
    ArrayPointerType initialize_pointer_by_copying(const AlgebraicVector &other) const;
};

#include "algebraiccontainers_algebraicmatrix.hpp"
#include "algebraiccontainers_algebraicvector.hpp"

#endif
