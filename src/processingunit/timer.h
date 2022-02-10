#ifndef FILE_TIMER
#define FILE_TIMER

#include <omp.h>

#include <chrono>
#include <memory>

#ifndef CPU_ONLY
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <device_launch_parameters.h>
#endif

class TimerDevice;
using Timer = std::unique_ptr<TimerDevice>;

class TimerDevice {
public:
    using timepoint = float;
    using timespan_sec = float;

    TimerDevice() = default;
    virtual ~TimerDevice();

    virtual void start() = 0;
    virtual void stop() = 0;
    virtual timespan_sec elapsedTime() const = 0;
};

class CHRONO_Timer : public TimerDevice {
public:
    using chronoClock = std::chrono::system_clock;
    using chronoTimespan = TimerDevice::timespan_sec;
    using chronoTimepoint = std::chrono::time_point<chronoClock>;

    CHRONO_Timer();
    ~CHRONO_Timer() override = default;

    void start() override;
    void stop() override;
    chronoTimespan elapsedTime() const override;

private:
    chronoTimepoint _startTime{};
    chronoTimepoint _stopTime{};
};

class OMP_Timer : public TimerDevice {
public:
    using ompTimepoint = TimerDevice::timepoint;
    using ompTimespan = TimerDevice::timespan_sec;

    OMP_Timer();
    ~OMP_Timer() override = default;

    void start() override;
    void stop() override;
    ompTimespan elapsedTime() const override;

private:
    ompTimepoint _startTime;
    ompTimepoint _stopTime;
};

#ifndef CPU_ONLY
class GPU_Timer : public TimerDevice {
public:
    using gpuEvent = cudaEvent_t;
    using gpuTimespan = TimerDevice::timespan_sec;
    using gpuTimetype = gpuTimespan;

    GPU_Timer();
    ~GPU_Timer() override;

    GPU_Timer(const GPU_Timer&) = delete;
    GPU_Timer(GPU_Timer&&) = delete;
    GPU_Timer& operator=(const GPU_Timer&) = delete;
    GPU_Timer& operator=(GPU_Timer&&) = delete;

    void start() override;
    void stop() override;

    gpuTimespan elapsedTime() const override;


private:
    gpuEvent initializeEvent();

    gpuEvent _startEvent;
    gpuEvent _stopEvent;
};
#endif

#endif
