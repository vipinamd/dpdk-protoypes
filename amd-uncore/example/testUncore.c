#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <sys/queue.h>

#include <rte_memory.h>
#include <rte_launch.h>
#include <rte_eal.h>
#include <rte_per_lcore.h>
#include <rte_lcore.h>
#include <rte_debug.h>

#include <rte_power_uncore.h>

#include <avx.h>

#define INTELTEST 0
#define AVX2TEST 0
#define AVX512TEST 1

int freqUpDown = 0;

/* Launch a function on lcore. 8< */
static int
lcore_hello(__rte_unused void *arg)
{
        unsigned lcore_id;
        int ret = 0;
        enum power_management_env env;

        lcore_id = rte_lcore_id();
        printf("hello from core %u\n", lcore_id);

        for (uint16_t i = 0; i < 100; i++)
        {
#ifdef INTELTEST
                avxIntelReference();
#endif

#ifdef AVX2TEST
                test256();
#endif

#ifdef AVX512TEST
                test512();
#endif
        }

        return 0;
}

int
main(int argc, char **argv)
{
        int ret;
        unsigned lcore_id;

        ret = rte_eal_init(argc, argv);
        if (ret < 0)
                rte_panic("Cannot init EAL\n");

        argc -= ret;
        argv += ret;

        if (argc == 2) {
                int usrFreq = atoi (argv[1]);
                freqUpDown = (usrFreq >= 0 && usrFreq < 2) ? usrFreq : 0;
        }

        ret = rte_power_uncore_init(rte_lcore_id (), 0);
        if (ret != 0)
                rte_panic("failed to initialize uncore!\n");

        lcore_id = 0;
        RTE_LCORE_FOREACH_WORKER(lcore_id) {
                switch (freqUpDown)
                {
                        case 1:
                                ret = rte_power_uncore_freq_max (lcore_id, 0);
                                if (ret < 0)
                                        rte_panic("failed to set max uncore for lcore %u!\n", lcore_id);
                                break;
                        case 2:
                                ret = rte_power_uncore_freq_min (lcore_id, 0);
                                if (ret < 0)
                                        rte_panic("failed to set min uncore for lcore %u!\n", lcore_id);
                                break;
                }
        }

        lcore_id = 0;
                rte_panic("Cannot init EAL\n");

        argc -= ret;
        argv += ret;

        if (argc == 2) {
                int usrFreq = atoi (argv[1]);
                freqUpDown = (usrFreq >= 0 && usrFreq < 2) ? usrFreq : 0;
        }

        ret = rte_power_uncore_init(rte_lcore_id (), 0);
        if (ret != 0)
                rte_panic("failed to initialize uncore!\n");

        lcore_id = 0;
        RTE_LCORE_FOREACH_WORKER(lcore_id) {
                switch (freqUpDown)
                {
                        case 1:
                                ret = rte_power_uncore_freq_max (lcore_id, 0);
                                if (ret < 0)
                                        rte_panic("failed to set max uncore for lcore %u!\n", lcore_id);
                                break;
                        case 2:
                                ret = rte_power_uncore_freq_min (lcore_id, 0);
                                if (ret < 0)
                                        rte_panic("failed to set min uncore for lcore %u!\n", lcore_id);
                                break;
                }
        }

        lcore_id = 0;
        RTE_LCORE_FOREACH_WORKER(lcore_id) {
                rte_eal_remote_launch(lcore_hello, NULL, lcore_id);
        }

        /* call it on main lcore too */
        lcore_hello(NULL);

        rte_eal_mp_wait_lcore();

        /* clean up the EAL */
        rte_eal_cleanup();

        return 0;
}
