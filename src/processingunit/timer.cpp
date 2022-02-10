#include "timer.h"

TimerDevice::~TimerDevice()
{}

// ###############################################################

CHRONO_Timer::CHRONO_Timer()
        : _startTime(chronoClock::now())
{}

void CHRONO_Timer::start()
{
    _startTime = chronoClock::now();
    _stopTime = chronoTimepoint{};
    return;
}

void CHRONO_Timer::stop()
{
    _stopTime = chronoClock::now();
    return;
}

CHRONO_Timer::chronoTimespan CHRONO_Timer::elapsedTime() const
{
    return std::chrono::duration<chronoTimespan>(_stopTime - _startTime).count();
}

// ###############################################################

OMP_Timer::OMP_Timer()
: _startTime(static_cast<ompTimepoint>(omp_get_wtime())), _stopTime(0)
{}

void OMP_Timer::start()
{
    _startTime = static_cast<ompTimepoint>(omp_get_wtime());
    _stopTime = -1;
    return;
}

void OMP_Timer::stop()
{
    _stopTime = static_cast<ompTimepoint>(omp_get_wtime());
    return;
}

OMP_Timer::ompTimespan OMP_Timer::elapsedTime() const
{
    return _stopTime - _startTime;
}

// ###############################################################

#ifndef CPU_ONLY
// public:
GPU_Timer::GPU_Timer()
: _startEvent(initializeEvent()),  _stopEvent(initializeEvent())
{}

GPU_Timer::~GPU_Timer()
{
    cudaEventDestroy(_startEvent);
    cudaEventDestroy(_stopEvent);
}

void GPU_Timer::start()
{
    cudaEventRecord(_startEvent);
    return;
}

void GPU_Timer::stop()
{
    cudaEventRecord(_stopEvent);
    return;
}

GPU_Timer::gpuTimespan GPU_Timer::elapsedTime() const
{
    gpuTimetype conversionFactor = static_cast<gpuTimetype>(1/1000.0);
    gpuTimetype elapsedSeconds;

    cudaEventSynchronize(_stopEvent);
    cudaEventElapsedTime(&elapsedSeconds, _startEvent, _stopEvent);
    return conversionFactor*elapsedSeconds;
}


//private:
GPU_Timer::gpuEvent GPU_Timer::initializeEvent()
{
    gpuEvent event;
    cudaEventCreate(&event);
    return event;
}
#endif