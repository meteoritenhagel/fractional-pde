template<class floating>
ProcessingUnitDevice<floating>::ProcessingUnitDevice()
{
    static_assert(isFloat() || isDouble(), "Template type for processing unit has to be either float or double!");
}

template<class floating>
constexpr bool ProcessingUnitDevice<floating>::isFloat()
{
    return std::is_same<floating, float>::value;
}

template<class floating>
constexpr bool ProcessingUnitDevice<floating>::isDouble()
{
    return std::is_same<floating, double>::value;
}
