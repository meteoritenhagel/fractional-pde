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

/**
 * Class TimerDevice serves as an abstract interface for time stopping purposes.
 * This is necessary, since e.g. CUDA has its own time stopping functions and the
 * use of stopping CUDA applications with CPU-only timers might result in unexpected behavior.
 */
class TimerDevice {
public:
    using timepoint = float;
    using timespan_sec = float;

    /**
     * Constructor
     */
    TimerDevice() = default;

    /**
     * Destructor
     */
    virtual ~TimerDevice();

    /**
     * Start the timer.
     */
    virtual void start() = 0;

    /**
     * Stop the timer.
     */
    virtual void stop() = 0;

    /**
     * Returns the elapsed time between calls of start() and stop()
     * @return the elapsed time
     */
    virtual timespan_sec elapsedTime() const = 0;
};

/**
 * Class CHRONO_Timer is a stop watch for measuring time elapsed in CPU-only processes.
 * It is based on std::chrono::system_clock.
 */
class CHRONO_Timer : public TimerDevice {
public:
    using chronoClock = std::chrono::system_clock;
    using chronoTimespan = TimerDevice::timespan_sec;
    using chronoTimepoint = std::chrono::time_point<chronoClock>;

    /**
     * Constructor
     */
    CHRONO_Timer();

    /**
     * @copydoc TimerDevice::~TimerDevice()
     */
    ~CHRONO_Timer() override = default;

    /**
    * @copydoc TimerDevice::start()
    */
    void start() override;

    /**
    * @copydoc TimerDevice::stop()
    */
    void stop() override;

    /**
    * @copydoc TimerDevice::elapsedTime() const
    */
    chronoTimespan elapsedTime() const override;

private:
    chronoTimepoint _startTime{}; //!< time point representing the starting point
    chronoTimepoint _stopTime{}; //!< time point representing the stopping point
};

/**
 * Class OMP_Timer is a stop watch for measuring time elapsed in parallelized CPU-only processes.
 * It is based on the OpenMP timing tools.
 */
class OMP_Timer : public TimerDevice {
public:
    using ompTimepoint = TimerDevice::timepoint;
    using ompTimespan = TimerDevice::timespan_sec;

    /**
     * Constructor
     */
    OMP_Timer();

    /**
     * @copydoc TimerDevice::~TimerDevice()
     */
    ~OMP_Timer() override = default;

    /**
     * @copydoc TimerDevice::start()
     */
    void start() override;

    /**
     * @copydoc TimerDevice::stop()
     */
    void stop() override;

    /**
     * @copydoc TimerDevice::elapsedTime() const
     */
    ompTimespan elapsedTime() const override;

private:
    ompTimepoint _startTime; //!< time point representing the starting point
    ompTimepoint _stopTime; //!< time point representing the stopping point
};

#ifndef CPU_ONLY
class GPU_Timer : public TimerDevice {
public:
    using gpuEvent = cudaEvent_t;
    using gpuTimespan = TimerDevice::timespan_sec;
    using gpuTimetype = gpuTimespan;

    /**
     * Constructor
     */
    GPU_Timer();

    /**
     * @copydoc TimerDevice::~TimerDevice()
     */
    ~GPU_Timer() override;

    /**
     * @copydoc TimerDevice::start()
     */
    void start() override;

    /**
     * @copydoc TimerDevice::stop()
     */
    void stop() override;

    /**
     * @copydoc TimerDevice::elapsedTime() const
     */
    gpuTimespan elapsedTime() const override;


private:
    gpuEvent _startEvent; //!< event representing the starting point of time measurement
    gpuEvent _stopEvent; //!< event representing the stopping point of time measurement

    /**
     * Creates a cudaEvent_t instance.
     * @return CUDA event
     */
    gpuEvent initializeEvent();

    GPU_Timer(const GPU_Timer&) = delete;
    GPU_Timer(GPU_Timer&&) = delete;
    GPU_Timer& operator=(const GPU_Timer&) = delete;
    GPU_Timer& operator=(GPU_Timer&&) = delete;
};
#endif

#endif
