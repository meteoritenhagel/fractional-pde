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
    assert(A.getNelems() == B.getNelems() && "ERROR: Dimension mismatch.");
    assert(typeid(*A.get_processing_unit()) == typeid(*B.get_processing_unit()) && "Processing Units must be identical.");

    return A.get_processing_unit()->xdot(A.getNelems(), A.data(), 1, B.data(), 1);
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
AlgebraicMatrix<floating>::AlgebraicMatrix(const ProcessingUnit<floating>& processingUnit, const MatrixDataType& A)
: _colMatrixFactory(processingUnit), _A(A), _Ainv(nullptr), _ipiv(0, 0, get_processing_unit()->get_memory_manager()), _arrayOfColumns(initializeArrayOfColumns())
{
}

template<class floating>
AlgebraicMatrix<floating>::AlgebraicMatrix(const AlgebraicMatrix &other)
: _colMatrixFactory(other._colMatrixFactory), _A(other._A), _Ainv(nullptr), _ipiv(other._ipiv), _arrayOfColumns(initializeArrayOfColumns())
{
    assert(typeid(*this->get_processing_unit()) == typeid(*other.get_processing_unit()) && "Processing Units must be identical.");
    // copy _Ainv
    if (other._Ainv)
    {
        _Ainv = std::make_unique<AlgebraicMatrix<floating>>(*other._Ainv);
    }
}

template<class floating>
void AlgebraicMatrix<floating>::moveTo(const ProcessingUnit<floating> processingUnit)
{
    _colMatrixFactory = ContainerFactory<floating>(processingUnit);

    if (_Ainv)
        _Ainv->moveTo(processingUnit);

    _A.moveTo(processingUnit->get_memory_manager());

    _ipiv.moveTo(processingUnit->get_memory_manager());
    resetArrayOfColumns();
}

template<class floating>
AlgebraicMatrix<floating>& AlgebraicMatrix<floating>::operator=(const AlgebraicMatrix &other)
{
    assert(typeid(*this->get_processing_unit()) == typeid(*other.get_processing_unit()) && "Processing Units must be identical.");
    if (this != &other)
    {
        _colMatrixFactory = other._colMatrixFactory;
        _A = other._A;
        _ipiv = DeviceArray<int>(0, 0, get_processing_unit()->get_memory_manager());
        _arrayOfColumns = initializeArrayOfColumns();

        // copy _Ainv
        if (other._Ainv)
        {
            _Ainv = std::make_unique<AlgebraicMatrix<floating>>(*other._Ainv);
            _ipiv = other._ipiv;
        }
        else
        {
            _Ainv = nullptr;
        }
    }

    return *this;
}

template<class floating>
AlgebraicMatrix<floating>& AlgebraicMatrix<floating>::resize(const SizeType nrows, const SizeType ncols)
{
    // TODO: Check if other members have to be reset
    _A.resize(nrows, ncols);
    resetArrayOfColumns();
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
    return _arrayOfColumns[col];
}

template<class floating>
AlgebraicVector<floating> const & AlgebraicMatrix<floating>::operator[](const SizeType col) const
{
    return _arrayOfColumns[col];
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
    const SizeType N = getNcols();
    auto row = *get_container_factory().createColumn(N);
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
    std::vector<floating> retvec(strtptr, strtptr + getNelems());
    return retvec;
}

template<class floating>
AlgebraicMatrix<floating> AlgebraicMatrix<floating>::flat() const
{
    auto copy(*this);
    copy.resize(getNelems(), 1);
    return copy;
}

template<class floating>
typename AlgebraicMatrix<floating>::SizeType AlgebraicMatrix<floating>::getNrows() const
{
    return _A.getN();
}

template<class floating>
typename AlgebraicMatrix<floating>::SizeType AlgebraicMatrix<floating>::getNcols() const
{
    return _A.getM();
}

template<class floating>
typename AlgebraicMatrix<floating>::SizeType AlgebraicMatrix<floating>::getNelems() const
{
    return _A.size();
}

template<class floating>
bool AlgebraicMatrix<floating>::isSquare() const
{
    return getNrows() == getNcols();
}

template<class floating>
ContainerFactory<floating> AlgebraicMatrix<floating>::get_container_factory() const
{
    return _colMatrixFactory;
}

template<class floating>
ProcessingUnit<floating> AlgebraicMatrix<floating>::get_processing_unit() const
{
    return _colMatrixFactory.getProcessingUnit();
}

template<class floating>
AlgebraicMatrix<floating> const & AlgebraicMatrix<floating>::getInverse() const
{
    return accessInverse();
}

template<class floating>
floating AlgebraicMatrix<floating>::getEuclidean() const
{
    const floating sum = scalarProduct(*this, *this);
    return sqrt(sum);
}

template<class floating>
floating AlgebraicMatrix<floating>::getMaximum() const
{
    const auto index = get_processing_unit()->ixamax(getNelems(), data(), 1);
    DeviceScalar<floating> maximum(data()[index], get_processing_unit()->get_memory_manager());
    maximum.moveTo(std::make_shared<CPU_Manager>());
    return maximum.value();
}

template<class floating>
std::string AlgebraicMatrix<floating>::display(std::string name) const
{
    return _A.display(name);
}

template<class floating>
AlgebraicMatrix<floating>& AlgebraicMatrix<floating>::add(AlgebraicMatrix<floating> const &B, floating const alpha)
{
    assert( this->getNcols() == B.getNcols() ); // identical dimensions?
    assert( this->getNrows() == B.getNrows() );
    assert(typeid(*this->get_processing_unit()) == typeid(*B.get_processing_unit()) && "Processing Units must be identical.");

    AlgebraicMatrix<floating> C = *this;

    unsigned int const N = this->getNelems();
    floating const * const b_arraystart = B.data();
    floating * const c_arraystart = this->data();
    unsigned int const inc = 1;

    get_processing_unit()->xaxpy(N, alpha, b_arraystart, inc, c_arraystart, inc);

    resetInverse();
    return *this;
}

template<class floating>
void AlgebraicMatrix<floating>::updateAdd(const SizeType col_begin, const SizeType col_end, const AlgebraicMatrix &B)
{
    assert(this->getNrows() == B.getNrows());
    assert(typeid(*this->get_processing_unit()) == typeid(*B.get_processing_unit()) && "Processing Units must be identical.");

    assert((col_end - col_begin) <= B.getNcols());
    for (SizeType j = 0; j < col_end - col_begin; ++j)
    {
        (*this)[col_begin + j] += B[j];
        //for (SizeType i = 0; i < getNrows(); ++i)
        //{
        //    (*this)(i, col_begin + j)+= B(i,j);
        //}
    }

    resetInverse();
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
    unsigned int const N = this->getNelems();
    floating* arraystart = this->data();
    unsigned int const incx = 1; // spacing between elements = 1

    get_processing_unit()->xscal(N, alpha, arraystart, incx);

    resetInverse();
    return;
}

template<class floating>
AlgebraicMatrix<floating> AlgebraicMatrix<floating>::mult(const AlgebraicMatrix &B) const
{
    assert( this->getNcols() == B.getNrows() ); // inner dimensions equal?
    assert(typeid(*this->get_processing_unit()) == typeid(*B.get_processing_unit()) && "Processing Units must be identical.");

    auto C = *get_container_factory().createMatrix(this->getNrows(), B.getNcols());

    floating const *const pA = this->data();
    floating const *const pB = B.data();
    floating *const pC = C.data();

    floating const alpha = 1.0;
    floating const beta = 0.0;

    unsigned int const M = C.getNrows();
    unsigned int const N = C.getNcols();
    unsigned int const K = this->getNcols();
    unsigned int const LDA = M;
    unsigned int const LDB = K;
    unsigned int const LDC = M;

    get_processing_unit()->xgemm(OperationType::Identical, OperationType::Identical,
                                 M, N, K,
                                 alpha, pA, LDA,
                                 pB, LDB,
                                 beta, pC, LDC);

    resetInverse();
    return C;
}

template<class floating>
AlgebraicVector<floating> AlgebraicMatrix<floating>::mult(const AlgebraicVector<floating> &u) const
{
    assert(this->getNcols() == u.size()); // #columns in matrix =? #elements in vector
    assert(typeid(*this->get_processing_unit()) == typeid(*u.get_processing_unit()) && "Processing Units must be identical.");

    auto res = *get_container_factory().createColumn(this->getNrows());

    floating const *const pA = this->data();
    floating const *const pu = u.data();           // pointer to memory of input vector
    floating *const      pf = res.data();         // pointer to memory of output vector

    floating const alpha = 1.0;
    floating const beta = 0.0;
    unsigned int const M = this->getNrows();
    unsigned int const N = this->getNcols();
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
AlgebraicVector<floating> AlgebraicMatrix<floating>::operator*(AlgebraicVector<floating> const &u) const
{
    assert(typeid(*this->get_processing_unit()) == typeid(*u.get_processing_unit()) && "Processing Units must be identical.");
    return mult(u);
}

template<class floating>
AlgebraicMatrix<floating> AlgebraicMatrix<floating>::operator/(const AlgebraicMatrix<floating> &B) const
{
    auto copied(B);
    //AlgebraicMatrix<floating> output = *get_container_factory().createMatrix(getNrows(), getNcols(), 0.0);
    this->invTimes(copied);

    return copied;
}

template<class floating>
void AlgebraicMatrix<floating>::invTimes(AlgebraicMatrix<floating> &B) const
{
    assert( this->getNrows() == B.getNrows() );
    assert(typeid(*this->get_processing_unit()) == typeid(*B.get_processing_unit()) && "Processing Units must be identical.");

    auto buffer = *get_container_factory().createMatrix(this->getNrows(), B.getNcols());
    
    auto &Ainv = accessInverse();

    floating const *const pA = Ainv.data();
    floating const *const pB = B.data();
    floating *const pC = buffer.data();

    floating const alpha = 1.0;
    floating const beta = 0.0;

    unsigned int const M = buffer.getNrows();
    unsigned int const N = buffer.getNcols();
    unsigned int const K = this->getNcols();
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

    getProcessingUnit()->xgetrs(OperationType::Identical, &N, &K, Ainv.data(), &LDA, _ipiv.data(), B.data(), &N, &info);
#endif
}

template<class floating>
AlgebraicMatrix<floating> AlgebraicMatrix<floating>::operator/(const std::vector<floating> x) const
{
    return *this / *_colMatrixFactory.createMatrix(this->getNrows(), x);
}

template<class floating>
AlgebraicVector<floating> AlgebraicMatrix<floating>::operator/(const AlgebraicVector<floating> &x) const
{
    AlgebraicVector<floating> output = *get_container_factory().createColumn(x.size(), 0.0);
    this->invTimes(x, output);
    
    return output;
}

template<class floating>
void AlgebraicMatrix<floating>::invTimes(const AlgebraicVector<floating> &x, AlgebraicVector<floating> &output) const
{
    assert(this->getNcols() == x.size()); // #columns in matrix =? #elements in vector
    assert(typeid(*this->get_processing_unit()) == typeid(*x.get_processing_unit()) && "Processing Units must be identical.");

#ifndef PLU
    auto &Ainv = accessInverse();

    floating const *const pA = Ainv.data();
    floating const *const pu = x.data();           // pointer to memory of input vector
    floating *const      pf = output.data();         // pointer to memory of output vector

    floating const alpha = 1.0;
    floating const beta = 0.0;
    unsigned int const M = this->getNrows();
    unsigned int const N = this->getNcols();
    unsigned int const LDA = N;

    get_processing_unit()->xgemv(OperationType::Identical, M, N, alpha, pA, LDA, pu, 1, beta, pf, 1);
#else
    const auto &Ainv = accessInverse();
    output = x;

    int N = Ainv.getNrows();
    int NRHS = 1;

    int LDA = N;

    int info(0);

    getProcessingUnit()->xgetrs(OperationType::Identical, &N, &NRHS, Ainv.data(), &LDA, _ipiv.data(), output.data(), &N, &info);

    if (info != 0) {
        std::cout << "No success (" << info << ") in dgetrs! " << __FILE__ << ":" << __LINE__ << std::endl;
        std::cout << "See http://www.netlib.org/lapack/explore-html/dd/d9a/group__double_g_ecomputational_ga58e332cb1b8ab770270843221a48296d.html#ga58e332cb1b8ab770270843221a48296d   ";
        std::cout << "for more information on INFO" << std::endl;
    }
#endif
}

// private:

template<class floating>
typename AlgebraicMatrix<floating>::ArrayOfColumns AlgebraicMatrix<floating>::initializeArrayOfColumns() const
{
    AlgebraicMatrix<floating>::ArrayOfColumns arrayOfColumns(getNcols());
    for (SizeType i = 0; i < getNcols(); ++i)
    {
        arrayOfColumns[i] = AlgebraicVector<floating>(get_processing_unit(), _A.getPointerToColumn(i));
    }
    return arrayOfColumns;
}

template<class floating>
void AlgebraicMatrix<floating>::resetArrayOfColumns()
{
    _arrayOfColumns = initializeArrayOfColumns();
}

template<class floating>
AlgebraicMatrix<floating>& AlgebraicMatrix<floating>::accessInverse()
{
    if (!isInverseSet())
    {
        recalculateInverse();
    }
    return *_Ainv;
}

template<class floating>
const AlgebraicMatrix<floating>& AlgebraicMatrix<floating>::accessInverse() const
{
    if (!isInverseSet())
    {
        recalculateInverse();
    }
    return *_Ainv;
}

template<class floating>
bool AlgebraicMatrix<floating>::isInverseSet() const
{
    return (_Ainv != nullptr) ? true : false;
}

template<class floating>
void AlgebraicMatrix<floating>::recalculateInverse() const
{
#ifndef PLU
    _Ainv = get_container_factory().createMatrix(getNrows(), getNcols(), 0.0);
    auto &eye = *_Ainv;
    assert(eye.isSquare());
    
    auto Afactorization = *std::make_unique<AlgebraicMatrix<floating>>(*this);
    
    int N = eye.getNrows();
    
    initializeIdentityMatrix(get_processing_unit()->get_memory_manager(), eye.data(), eye.getNrows(), eye.getNcols());

    int LDA = N;

    _ipiv = DeviceArray<int>(N, 0, get_processing_unit()->get_memory_manager());
    int info(0);

    get_processing_unit()->xgetrf(&N, &N, Afactorization.data(), &LDA, _ipiv.data(), &info);

    if (info != 0) {
        std::cout << "No success (" << info << ") in dgetrf! " << __FILE__ << ":" << __LINE__ << std::endl;
        std::cout << "See  http://www.netlib.org/lapack/explore-html/dd/d9a/group__double_g_ecomputational_ga0019443faea08275ca60a734d0593e60.html#ga0019443faea08275ca60a734d0593e60   ";
        std::cout << "for more information on INFO" << std::endl;
    }

    get_processing_unit()->xgetrs(OperationType::Identical, &N, &N, Afactorization.data(), &LDA, _ipiv.data(), eye.data(), &LDA, &info);
#else
    _Ainv = std::make_unique<AlgebraicMatrix<floating>>(*this);

    AlgebraicMatrix<floating> &Ainv = *_Ainv;
    assert(Ainv.isSquare());

    int N = Ainv.getNrows();
    int LDA = N;

    _ipiv = DeviceArray<int>(N, 0, getProcessingUnit()->get_memory_manager());
    int info(0);

    getProcessingUnit()->xgetrf(&N, &N, Ainv.data(), &LDA, _ipiv.data(), &info);

    if (info != 0) {
        std::cout << "No success (" << info << ") in dgetrf! " << __FILE__ << ":" << __LINE__ << std::endl;
        std::cout << "See  http://www.netlib.org/lapack/explore-html/dd/d9a/group__double_g_ecomputational_ga0019443faea08275ca60a734d0593e60.html#ga0019443faea08275ca60a734d0593e60   ";
        std::cout << "for more information on INFO" << std::endl;
    }
#endif
}

template<class floating>
void AlgebraicMatrix<floating>::resetInverse() const
{
    _Ainv.reset();
}
#endif
