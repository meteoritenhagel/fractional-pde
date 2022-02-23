#include "timer.h"

TimerDevice::~TimerDevice()
{}

// ###############################################################

ChronoTimer::ChronoTimer()
        : _start_time(ChronoClock::now())
{}

void ChronoTimer::start()
{
    _start_time = ChronoClock::now();
    _stop_time = ChronoTimepoint{};
    return;
}

void ChronoTimer::stop()
{
    _stop_time = ChronoClock::now();
    return;
}

ChronoTimer::ChronoTimespan ChronoTimer::elapsed_time() const
{
    return std::chrono::duration<ChronoTimespan>(_stop_time - _start_time).count();
}

// ###############################################################

OmpTimer::OmpTimer()
: _start_time(static_cast<OmpTimepoint>(omp_get_wtime())), _stop_time(0)
{}

void OmpTimer::start()
{
    _start_time = static_cast<OmpTimepoint>(omp_get_wtime());
    _stop_time = -1;
    return;
}

void OmpTimer::stop()
{
    _stop_time = static_cast<OmpTimepoint>(omp_get_wtime());
    return;
}

OmpTimer::OmpTimespan OmpTimer::elapsed_time() const
{
    return _stop_time - _start_time;
}

// ###############################################################

#ifndef CPU_ONLY
// public:
GpuTimer::GpuTimer()
: _start_event(initialize_event()), _stop_event(initialize_event())
{}

GpuTimer::~GpuTimer()
{
    cudaEventDestroy(_start_event);
    cudaEventDestroy(_stop_event);
}

void GpuTimer::start()
{
    cudaEventRecord(_start_event);
    return;
}

void GpuTimer::stop()
{
    cudaEventRecord(_stop_event);
    return;
}

GpuTimer::GpuTimespan GpuTimer::elapsed_time() const
{
    GpuTimetype conversionFactor = static_cast<GpuTimetype>(1 / 1000.0);
    GpuTimetype elapsedSeconds;

    cudaEventSynchronize(_stop_event);
    cudaEventElapsedTime(&elapsedSeconds, _start_event, _stop_event);
    return conversionFactor*elapsedSeconds;
}


//private:
GpuTimer::GpuEvent GpuTimer::initialize_event()
{
    GpuEvent event;
    cudaEventCreate(&event);
    return event;
}
#endif