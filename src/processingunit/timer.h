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
 * use of stopping CUDA applications with Cpu-only timers might result in unexpected behavior.
 */
class TimerDevice {
public:
    using TimePoint = float;
    using TimespanSec = float;

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
    virtual TimespanSec elapsed_time() const = 0;
};

/**
 * Class ChronoTimer is a stop watch for measuring time elapsed in Cpu-only processes.
 * It is based on std::chrono::system_clock.
 */
class ChronoTimer : public TimerDevice {
public:
    using ChronoClock = std::chrono::system_clock;
    using ChronoTimespan = TimerDevice::TimespanSec;
    using ChronoTimepoint = std::chrono::time_point<ChronoClock>;

    /**
     * Constructor
     */
    ChronoTimer();

    /**
     * @copydoc TimerDevice::~TimerDevice()
     */
    ~ChronoTimer() override = default;

    /**
    * @copydoc TimerDevice::start()
    */
    void start() override;

    /**
    * @copydoc TimerDevice::stop()
    */
    void stop() override;

    /**
    * @copydoc TimerDevice::elapsed_time() const
    */
    ChronoTimespan elapsed_time() const override;

private:
    ChronoTimepoint _start_time{}; //!< time point representing the starting point
    ChronoTimepoint _stop_time{}; //!< time point representing the stopping point
};

/**
 * Class OmpTimer is a stop watch for measuring time elapsed in parallelized Cpu-only processes.
 * It is based on the OpenMP timing tools.
 */
class OmpTimer : public TimerDevice {
public:
    using OmpTimepoint = TimerDevice::TimePoint;
    using OmpTimespan = TimerDevice::TimespanSec;

    /**
     * Constructor
     */
    OmpTimer();

    /**
     * @copydoc TimerDevice::~TimerDevice()
     */
    ~OmpTimer() override = default;

    /**
     * @copydoc TimerDevice::start()
     */
    void start() override;

    /**
     * @copydoc TimerDevice::stop()
     */
    void stop() override;

    /**
     * @copydoc TimerDevice::elapsed_time() const
     */
    OmpTimespan elapsed_time() const override;

private:
    OmpTimepoint _start_time; //!< time point representing the starting point
    OmpTimepoint _stop_time; //!< time point representing the stopping point
};

#ifndef CPU_ONLY
class GpuTimer : public TimerDevice {
public:
    using GpuEvent = cudaEvent_t;
    using GpuTimespan = TimerDevice::TimespanSec;
    using GpuTimetype = GpuTimespan;

    /**
     * Constructor
     */
    GpuTimer();

    /**
     * @copydoc TimerDevice::~TimerDevice()
     */
    ~GpuTimer() override;

    /**
     * @copydoc TimerDevice::start()
     */
    void start() override;

    /**
     * @copydoc TimerDevice::stop()
     */
    void stop() override;

    /**
     * @copydoc TimerDevice::elapsed_time() const
     */
    GpuTimespan elapsed_time() const override;


private:
    GpuEvent _start_event; //!< event representing the starting point of time measurement
    GpuEvent _stop_event; //!< event representing the stopping point of time measurement

    /**
     * Creates a cudaEvent_t instance.
     * @return CUDA event
     */
    GpuEvent initialize_event();

    GpuTimer(const GpuTimer&) = delete;
    GpuTimer(GpuTimer&&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;
    GpuTimer& operator=(GpuTimer&&) = delete;
};
#endif

#endif
