import main_sac_phase_0
import main_phase_0_plus
import main_sac_phase_1
import main_sac_phase_2

def main():
    # Phase 0: Basic navigation
   # main_sac_phase_0.main_0()

    # Phase 0.5 : policy turned deterministic due to a autotuned to 0.0001 so we make a = 0.2 fixed to keep exploring
    #main_phase_0_plus.main_0_plus()

    # Phase 1: Hazard avoidance
    #main_sac_phase_1.main_1()

    # Phase 2: Hazard avoidance advanced
    main_sac_phase_2.main_2()


if __name__ == "__main__":
    main()