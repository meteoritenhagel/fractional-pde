#ifndef FILE_COLMATRIX_TPP
#define FILE_COLMATRIX_TPP

#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include "containerfactory.h"

template<class floating>
AlgebraicMatrix<floating> operator* (const floating alpha, const AlgebraicMatrix<floating> &B)
{
    AlgebraicMatrix<floating> tmp(B);
    tmp.scale(alpha);
    return tmp;
}

template<class floating>
floating scalarProduct(const BlockVector<floating> &A, const BlockVector<floating> &B)
{
    assert(A.get_num_elements() == B.get_num_elements() && "ERROR: Dimension mismatch.");
    assert(typeid(*A.get_processing_unit()) == typeid(*B.get_processing_unit()) && "Processing Units must be identical.");

    return A.get_processing_unit()->xdot(A.get_num_elements(), A.data(), 1, B.data(), 1);
}

template<class floating>
floating scalarProduct(const AlgebraicVector<floating> &A, const AlgebraicVector<floating> &B)
{
    assert(A.size() == B.size() && "ERROR: Dimension mismatch.");
    assert(typeid(*A.get_processing_unit()) == typeid(*B.get_processing_unit()) && "Processing Units must be identical.");

    return A.get_processing_unit()->xdot(A.size(), A.data(), 1, B.data(), 1);
}

// public:

template<class floating>
AlgebraicMatrix<floating>::AlgebraicMatrix(const ProcessingUnit<floating>& processing_unit, const MatrixDataType& A)
: _container_factory(processing_unit), _A(A), _inverse(nullptr), _permutation(0, 0, get_processing_unit()->get_memory_manager()), _array_of_columns(
        initialize_array_of_columns())
{
}

template<class floating>
AlgebraicMatrix<floating>::AlgebraicMatrix(const AlgebraicMatrix &other)
: _container_factory(other._container_factory), _A(other._A), _inverse(nullptr), _permutation(other._permutation), _array_of_columns(
        initialize_array_of_columns())
{
    assert(typeid(*this->get_processing_unit()) == typeid(*other.get_processing_unit()) && "Processing Units must be identical.");
    // copy _inverse
    if (other._inverse)
    {
        _inverse = std::make_unique<AlgebraicMatrix<floating>>(*other._inverse);
    }
}

template<class floating>
void AlgebraicMatrix<floating>::move_to(const ProcessingUnit<floating> processing_unit)
{
    _container_factory = ContainerFactory<floating>(processing_unit);

    if (_inverse)
        _inverse->move_to(processing_unit);

    _A.move_to(processing_unit->get_memory_manager());

    _permutation.move_to(processing_unit->get_memory_manager());
    reset_array_of_columns();
}

template<class floating>
AlgebraicMatrix<floating>& AlgebraicMatrix<floating>::operator=(const AlgebraicMatrix &other)
{
    assert(typeid(*this->get_processing_unit()) == typeid(*other.get_processing_unit()) && "Processing Units must be identical.");
    if (this != &other)
    {
        _container_factory = other._container_factory;
        _A = other._A;
        _permutation = DeviceArray<int>(0, 0, get_processing_unit()->get_memory_manager());
        _array_of_columns = initialize_array_of_columns();

        // copy _inverse
        if (other._inverse)
        {
            _inverse = std::make_unique<AlgebraicMatrix<floating>>(*other._inverse);
            _permutation = other._permutation;
        }
        else
        {
            _inverse = nullptr;
        }
    }

    return *this;
}

template<class floating>
AlgebraicMatrix<floating>& AlgebraicMatrix<floating>::resize(const SizeType num_rows, const SizeType num_cols)
{
    // TODO: Check if other members have to be reset
    _A.resize(num_rows, num_cols);
    reset_array_of_columns();
    return *this;
}

template<class floating>
floating* AlgebraicMatrix<floating>::data()
{
    return _A.data();
}

template<class floating>
const floating* AlgebraicMatrix<floating>::data() const
{
    return _A.data();
}

template<class floating>
AlgebraicVector<floating>& AlgebraicMatrix<floating>::operator[](const SizeType col)
{
    return _array_of_columns[col];
}

template<class floating>
AlgebraicVector<floating> const & AlgebraicMatrix<floating>::operator[](const SizeType col) const
{
    return _array_of_columns[col];
}

template<class floating>
floating& AlgebraicMatrix<floating>::operator()(const SizeType i, const SizeType j)
{
    assert(typeid(*get_processing_unit()) == typeid(*std::make_shared<Cpu<floating>>()) && "Must be on Cpu");
    return _A(i,j);
}

template<class floating>
floating const & AlgebraicMatrix<floating>::operator()(const SizeType i, const SizeType j) const
{
    assert(typeid(*get_processing_unit()) == typeid(*std::make_shared<Cpu<floating>>()) && "Must be on Cpu");
    return _A(i,j);
}
template<class floating>

AlgebraicVector<floating> AlgebraicMatrix<floating>::getRow(const SizeType i) const
{
    assert(typeid(*get_processing_unit()) == typeid(*std::make_shared<Cpu<floating>>()) && "Must be on Cpu");
    const SizeType N = get_num_cols();
    auto row = *get_container_factory().create_array(N);
    for(SizeType j = 0; j < N; ++j)
    {
        row[j] = _A(i, j);
    }
    return row;
}


template<class floating>
std::vector<floating> AlgebraicMatrix<floating>::values() const
{
    floating const * const strtptr = this->data();
    std::vector<floating> retvec(strtptr, strtptr + get_num_elements());
    return retvec;
}

template<class floating>
AlgebraicMatrix<floating> AlgebraicMatrix<floating>::flat() const
{
    auto copy(*this);
    copy.resize(get_num_elements(), 1);
    return copy;
}

template<class floating>
typename AlgebraicMatrix<floating>::SizeType AlgebraicMatrix<floating>::get_num_rows() const
{
    return _A.get_num_rows();
}

template<class floating>
typename AlgebraicMatrix<floating>::SizeType AlgebraicMatrix<floating>::get_num_cols() const
{
    return _A.get_num_cols();
}

template<class floating>
typename AlgebraicMatrix<floating>::SizeType AlgebraicMatrix<floating>::get_num_elements() const
{
    return _A.size();
}

template<class floating>
bool AlgebraicMatrix<floating>::is_square() const
{
    return get_num_rows() == get_num_cols();
}

template<class floating>
ContainerFactory<floating> AlgebraicMatrix<floating>::get_container_factory() const
{
    return _container_factory;
}

template<class floating>
ProcessingUnit<floating> AlgebraicMatrix<floating>::get_processing_unit() const
{
    return _container_factory.get_processing_unit();
}

template<class floating>
AlgebraicMatrix<floating> const & AlgebraicMatrix<floating>::get_inverse() const
{
    return access_inverse();
}

template<class floating>
floating AlgebraicMatrix<floating>::get_euclidean_norm() const
{
    const floating sum = scalarProduct(*this, *this);
    return sqrt(sum);
}

template<class floating>
floating AlgebraicMatrix<floating>::get_maximum_norm() const
{
    const auto index = get_processing_unit()->ixamax(get_num_elements(), data(), 1);
    DeviceScalar<floating> maximum(data()[index], get_processing_unit()->get_memory_manager());
    maximum.move_to(std::make_shared<CpuManager>());
    return maximum.value();
}

template<class floating>
std::string AlgebraicMatrix<floating>::display(std::string name) const
{
    return _A.display(name);
}

template<class floating>
AlgebraicMatrix<floating>& AlgebraicMatrix<floating>::add(AlgebraicMatrix<floating> const &B, floating const scalar)
{
    assert(this->get_num_cols() == B.get_num_cols() ); // identical dimensions?
    assert(this->get_num_rows() == B.get_num_rows() );
    assert(typeid(*this->get_processing_unit()) == typeid(*B.get_processing_unit()) && "Processing Units must be identical.");

    AlgebraicMatrix<floating> C = *this;

    unsigned int const N = this->get_num_elements();
    floating const * const b_arraystart = B.data();
    floating * const c_arraystart = this->data();
    unsigned int const inc = 1;

    get_processing_unit()->xaxpy(N, scalar, b_arraystart, inc, c_arraystart, inc);

    reset_inverse();
    return *this;
}

template<class floating>
void AlgebraicMatrix<floating>::updateAdd(const SizeType col_begin, const SizeType col_end, const AlgebraicMatrix &B)
{
    assert(this->get_num_rows() == B.get_num_rows());
    assert(typeid(*this->get_processing_unit()) == typeid(*B.get_processing_unit()) && "Processing Units must be identical.");

    assert((col_end - col_begin) <= B.get_num_cols());
    for (SizeType j = 0; j < col_end - col_begin; ++j)
    {
        (*this)[col_begin + j] += B[j];
        //for (SizeType i = 0; i < get_num_rows(); ++i)
        //{
        //    (*this)(i, col_begin + j)+= B(i,j);
        //}
    }

    reset_inverse();
    return;
}

template<class floating>
AlgebraicMatrix<floating> AlgebraicMatrix<floating>::operator+(const AlgebraicMatrix &B) const
{
    auto tmp = *this;
    return tmp.add(B);
}

template<class floating>
AlgebraicMatrix<floating>& AlgebraicMatrix<floating>::operator+=(const AlgebraicMatrix &B)
{
    assert(typeid(*this->get_processing_unit()) == typeid(*B.get_processing_unit()) && "Processing Units must be identical.");
    return this->add(B);
}

template<class floating>
AlgebraicMatrix<floating> AlgebraicMatrix<floating>::operator-(const AlgebraicMatrix &B) const
{
    assert(typeid(*this->get_processing_unit()) == typeid(*B.get_processing_unit()) && "Processing Units must be identical.");
    auto tmp = *this;
    return tmp.add(B, -1.0);
}

template<class floating>
void AlgebraicMatrix<floating>::scale(floating const alpha)
{
    unsigned int const N = this->get_num_elements();
    floating* arraystart = this->data();
    unsigned int const incx = 1; // spacing between elements = 1

    get_processing_unit()->xscal(N, alpha, arraystart, incx);

    reset_inverse();
    return;
}

template<class floating>
AlgebraicMatrix<floating> AlgebraicMatrix<floating>::mult(const AlgebraicMatrix &B) const
{
    assert(this->get_num_cols() == B.get_num_rows() ); // inner dimensions equal?
    assert(typeid(*this->get_processing_unit()) == typeid(*B.get_processing_unit()) && "Processing Units must be identical.");

    auto C = *get_container_factory().create_matrix(this->get_num_rows(), B.get_num_cols());

    floating const *const pA = this->data();
    floating const *const pB = B.data();
    floating *const pC = C.data();

    floating const alpha = 1.0;
    floating const beta = 0.0;

    unsigned int const M = C.get_num_rows();
    unsigned int const N = C.get_num_cols();
    unsigned int const K = this->get_num_cols();
    unsigned int const LDA = M;
    unsigned int const LDB = K;
    unsigned int const LDC = M;

    get_processing_unit()->xgemm(OperationType::Identical, OperationType::Identical,
                                 M, N, K,
                                 alpha, pA, LDA,
                                 pB, LDB,
                                 beta, pC, LDC);

    reset_inverse();
    return C;
}

template<class floating>
AlgebraicVector<floating> AlgebraicMatrix<floating>::mult(const AlgebraicVector<floating> &vec) const
{
    assert(this->get_num_cols() == vec.size()); // #columns in matrix =? #elements in vector
    assert(typeid(*this->get_processing_unit()) == typeid(*vec.get_processing_unit()) && "Processing Units must be identical.");

    auto res = *get_container_factory().create_array(this->get_num_rows());

    floating const *const pA = this->data();
    floating const *const pu = vec.data();           // pointer to memory of input vector
    floating *const      pf = res.data();         // pointer to memory of output vector

    floating const alpha = 1.0;
    floating const beta = 0.0;
    unsigned int const M = this->get_num_rows();
    unsigned int const N = this->get_num_cols();
    unsigned int const LDA = N;

    get_processing_unit()->xgemv(OperationType::Identical, M, N, alpha, pA, LDA, pu, 1, beta, pf, 1);

    return res;
}

template<class floating>
AlgebraicMatrix<floating> AlgebraicMatrix<floating>::operator*(const AlgebraicMatrix &B) const
{
    assert(typeid(*this->get_processing_unit()) == typeid(*B.get_processing_unit()) && "Processing Units must be identical.");
    return mult(B);
}

template<class floating>
AlgebraicVector<floating> AlgebraicMatrix<floating>::operator*(AlgebraicVector<floating> const &vec) const
{
    assert(typeid(*this->get_processing_unit()) == typeid(*vec.get_processing_unit()) && "Processing Units must be identical.");
    return mult(vec);
}

template<class floating>
AlgebraicMatrix<floating> AlgebraicMatrix<floating>::operator/(const AlgebraicMatrix<floating> &B) const
{
    auto copied(B);
    //AlgebraicMatrix<floating> output = *get_container_factory().create_matrix(get_num_rows(), get_num_cols(), 0.0);
    this->invTimes(copied);

    return copied;
}

template<class floating>
void AlgebraicMatrix<floating>::invTimes(AlgebraicMatrix<floating> &B) const
{
    assert(this->get_num_rows() == B.get_num_rows() );
    assert(typeid(*this->get_processing_unit()) == typeid(*B.get_processing_unit()) && "Processing Units must be identical.");

    auto buffer = *get_container_factory().create_matrix(this->get_num_rows(), B.get_num_cols());
    
    auto &Ainv = access_inverse();

    floating const *const pA = Ainv.data();
    floating const *const pB = B.data();
    floating *const pC = buffer.data();

    floating const alpha = 1.0;
    floating const beta = 0.0;

    unsigned int const M = buffer.get_num_rows();
    unsigned int const N = buffer.get_num_cols();
    unsigned int const K = this->get_num_cols();
    unsigned int const LDA = M;
    unsigned int const LDB = K;
    unsigned int const LDC = M;

#ifndef PLU
    get_processing_unit()->xgemm(OperationType::Identical, OperationType::Identical,
                                 M, N, K,
                                 alpha, pA, LDA,
                                 pB, LDB,
                                 beta, pC, LDC);
    B = buffer;
#else
    int info(0);

    getProcessingUnit()->xgetrs(OperationType::Identical, &N, &K, Ainv.data(), &LDA, _permutation.data(), B.data(), &N, &info);
#endif
}

template<class floating>
AlgebraicMatrix<floating> AlgebraicMatrix<floating>::operator/(const std::vector<floating>& vec) const
{
    return *this / *_container_factory.create_matrix(this->get_num_rows(), vec);
}

template<class floating>
AlgebraicVector<floating> AlgebraicMatrix<floating>::operator/(const AlgebraicVector<floating> &vec) const
{
    AlgebraicVector<floating> output = *get_container_factory().create_array(vec.size(), 0.0);
    this->inverse_times(vec, output);
    
    return output;
}

template<class floating>
void AlgebraicMatrix<floating>::inverse_times(const AlgebraicVector<floating> &vec, AlgebraicVector<floating> &result) const
{
    assert(this->get_num_cols() == vec.size()); // #columns in matrix =? #elements in vector
    assert(typeid(*this->get_processing_unit()) == typeid(*vec.get_processing_unit()) && "Processing Units must be identical.");

#ifndef PLU
    auto &Ainv = access_inverse();

    floating const *const pA = Ainv.data();
    floating const *const pu = vec.data();           // pointer to memory of input vector
    floating *const      pf = result.data();         // pointer to memory of result vector

    floating const alpha = 1.0;
    floating const beta = 0.0;
    unsigned int const M = this->get_num_rows();
    unsigned int const N = this->get_num_cols();
    unsigned int const LDA = N;

    get_processing_unit()->xgemv(OperationType::Identical, M, N, alpha, pA, LDA, pu, 1, beta, pf, 1);
#else
    const auto &Ainv = accessInverse();
    result = vec;

    int N = Ainv.get_num_rows();
    int NRHS = 1;

    int LDA = N;

    int info(0);

    getProcessingUnit()->xgetrs(OperationType::Identical, &N, &NRHS, Ainv.data(), &LDA, _permutation.data(), result.data(), &N, &info);

    if (info != 0) {
        std::cout << "No success (" << info << ") in dgetrs! " << __FILE__ << ":" << __LINE__ << std::endl;
        std::cout << "See http://www.netlib.org/lapack/explore-html/dd/d9a/group__double_g_ecomputational_ga58e332cb1b8ab770270843221a48296d.html#ga58e332cb1b8ab770270843221a48296d   ";
        std::cout << "for more information on INFO" << std::endl;
    }
#endif
}

// private:

template<class floating>
typename AlgebraicMatrix<floating>::ArrayOfColumns AlgebraicMatrix<floating>::initialize_array_of_columns() const
{
    AlgebraicMatrix<floating>::ArrayOfColumns arrayOfColumns(get_num_cols());
    for (SizeType i = 0; i < get_num_cols(); ++i)
    {
        arrayOfColumns[i] = AlgebraicVector<floating>(get_processing_unit(), _A.get_pointer_to_column(i));
    }
    return arrayOfColumns;
}

template<class floating>
void AlgebraicMatrix<floating>::reset_array_of_columns()
{
    _array_of_columns = initialize_array_of_columns();
}

template<class floating>
AlgebraicMatrix<floating>& AlgebraicMatrix<floating>::access_inverse()
{
    if (!is_inverse_set())
    {
        recalculate_inverse();
    }
    return *_inverse;
}

template<class floating>
const AlgebraicMatrix<floating>& AlgebraicMatrix<floating>::access_inverse() const
{
    if (!is_inverse_set())
    {
        recalculate_inverse();
    }
    return *_inverse;
}

template<class floating>
bool AlgebraicMatrix<floating>::is_inverse_set() const
{
    return (_inverse != nullptr) ? true : false;
}

template<class floating>
void AlgebraicMatrix<floating>::recalculate_inverse() const
{
#ifndef PLU
    _inverse = get_container_factory().create_matrix(get_num_rows(), get_num_cols(), 0.0);
    auto &eye = *_inverse;
    assert(eye.is_square());
    
    auto Afactorization = *std::make_unique<AlgebraicMatrix<floating>>(*this);
    
    int N = eye.get_num_rows();

    initialize_identity_matrix(get_processing_unit()->get_memory_manager(), eye.data(), eye.get_num_rows(),
                               eye.get_num_cols());

    int LDA = N;

    _permutation = DeviceArray<int>(N, 0, get_processing_unit()->get_memory_manager());
    int info(0);

    get_processing_unit()->xgetrf(&N, &N, Afactorization.data(), &LDA, _permutation.data(), &info);

    if (info != 0) {
        std::cout << "No success (" << info << ") in dgetrf! " << __FILE__ << ":" << __LINE__ << std::endl;
        std::cout << "See  http://www.netlib.org/lapack/explore-html/dd/d9a/group__double_g_ecomputational_ga0019443faea08275ca60a734d0593e60.html#ga0019443faea08275ca60a734d0593e60   ";
        std::cout << "for more information on INFO" << std::endl;
    }

    get_processing_unit()->xgetrs(OperationType::Identical, &N, &N, Afactorization.data(), &LDA, _permutation.data(), eye.data(), &LDA, &info);
#else
    _inverse = std::make_unique<AlgebraicMatrix<floating>>(*this);

    AlgebraicMatrix<floating> &Ainv = *_inverse;
    assert(Ainv.isSquare());

    int N = Ainv.get_num_rows();
    int LDA = N;

    _permutation = DeviceArray<int>(N, 0, getProcessingUnit()->get_memory_manager());
    int info(0);

    getProcessingUnit()->xgetrf(&N, &N, Ainv.data(), &LDA, _permutation.data(), &info);

    if (info != 0) {
        std::cout << "No success (" << info << ") in dgetrf! " << __FILE__ << ":" << __LINE__ << std::endl;
        std::cout << "See  http://www.netlib.org/lapack/explore-html/dd/d9a/group__double_g_ecomputational_ga0019443faea08275ca60a734d0593e60.html#ga0019443faea08275ca60a734d0593e60   ";
        std::cout << "for more information on INFO" << std::endl;
    }
#endif
}

template<class floating>
void AlgebraicMatrix<floating>::reset_inverse() const
{
    _inverse.reset();
}
#endif
