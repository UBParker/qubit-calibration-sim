
# QUA library version: 1.4.1


from qm import CompilerOptionArguments
from qm.qua import *

with program() as prog:
    r1 = declare_output_stream("client", stream_id='r1')
    r2 = declare_output_stream("client", stream_id='r2')
    r3 = declare_output_stream("client", stream_id='r3')
    v1 = declare(int, )
    v2 = declare(int, )
    v3 = declare(fixed, )
    v4 = declare(fixed, )
    a1 = declare(int, value=[4, 316, 629, 941, 1254, 1566, 1879, 2191, 2504, 2816, 3128, 3441, 3753, 4066, 4378, 4691, 5003, 5316, 5628, 5941, 6253, 6565, 6878, 7190, 7503, 7815, 8128, 8440, 8753, 9065, 9378, 9690, 10002, 10315, 10627, 10940, 11252, 11565, 11877, 12190, 12502, 12814, 13127, 13439, 13752, 14064, 14377, 14689, 15002, 15314, 15626, 15939, 16251, 16564, 16876, 17189, 17501, 17814, 18126, 18439, 18751, 19063, 19376, 19688, 20001, 20313, 20626, 20938, 21251, 21563, 21876, 22188, 22500, 22813, 23125, 23438, 23750, 24063, 24375, 24688, 25000])
    with for_(v1,0,(v1<1000),(v1+1)):
        save(v1, r1)
        with for_each_((v2),(a1)):
            play('x90', 'qubit')
            wait(v2, 'qubit')
            play('x180', 'qubit')
            wait(v2, 'qubit')
            play('-x90', 'qubit')
            align('qubit', 'resonator')
            measure('readout', 'resonator', dual_demod.full("cos", "sin", v3), dual_demod.full("minus_sin", "cos", v4))
            save(v3, r2)
            save(v4, r3)
            wait(62500, 'resonator')
            align()
    with stream_processing():
        r1.save("n")
        r2.buffer(81).average().save("I")
        r3.buffer(81).average().save("Q")

config = {
    "controllers": {
        "con1": {
            "analog_outputs": {
                "1": {
                    "offset": 0.0,
                },
                "2": {
                    "offset": 0.0,
                },
                "3": {
                    "offset": 0.0,
                },
                "4": {
                    "offset": 0.0,
                },
            },
            "digital_outputs": {},
            "analog_inputs": {
                "1": {
                    "offset": 0.0011093127696236397,
                    "gain_db": 0,
                },
                "2": {
                    "offset": -0.0009993545058356748,
                    "gain_db": 0,
                },
            },
        },
    },
    "elements": {
        "qubit": {
            "mixInputs": {
                "I": ('con1', 1),
                "Q": ('con1', 2),
                "lo_frequency": 5000000000.0,
                "mixer": "mixer_qubit",
            },
            "intermediate_frequency": 123700073.93181705,
            "operations": {
                "x180": "x180_pulse",
                "x90": "x90_pulse",
                "-x90": "-x90_pulse",
                "y90": "y90_pulse",
                "saturation": "saturation_pulse",
            },
        },
        "resonator": {
            "mixInputs": {
                "I": ('con1', 3),
                "Q": ('con1', 4),
                "lo_frequency": 6950000000.0,
                "mixer": "mixer_resonator",
            },
            "intermediate_frequency": 90843700.88658619,
            "operations": {
                "readout": "readout_pulse",
            },
            "outputs": {
                "out1": ('con1', 1),
                "out2": ('con1', 2),
            },
            "time_of_flight": 236,
            "smearing": 0,
        },
    },
    "pulses": {
        "x180_pulse": {
            "operation": "control",
            "length": 40,
            "waveforms": {
                "I": "x180_wf",
                "Q": "zero_wf",
            },
        },
        "x90_pulse": {
            "operation": "control",
            "length": 40,
            "waveforms": {
                "I": "x90_wf",
                "Q": "zero_wf",
            },
        },
        "-x90_pulse": {
            "operation": "control",
            "length": 40,
            "waveforms": {
                "I": "minus_x90_wf",
                "Q": "zero_wf",
            },
        },
        "y90_pulse": {
            "operation": "control",
            "length": 40,
            "waveforms": {
                "I": "zero_wf",
                "Q": "x90_wf",
            },
        },
        "saturation_pulse": {
            "operation": "control",
            "length": 20000,
            "waveforms": {
                "I": "saturation_wf",
                "Q": "zero_wf",
            },
        },
        "readout_pulse": {
            "operation": "measurement",
            "length": 1000,
            "waveforms": {
                "I": "readout_wf",
                "Q": "zero_wf",
            },
            "integration_weights": {
                "cos": "cosine_weights",
                "sin": "sine_weights",
                "minus_sin": "minus_sine_weights",
                "rotated_cos": "rotated_cosine_weights",
                "rotated_sin": "rotated_sine_weights",
                "rotated_minus_sin": "rotated_minus_sine_weights",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "zero_wf": {
            "type": "constant",
            "sample": 0.0,
        },
        "saturation_wf": {
            "type": "constant",
            "sample": 0.002,
        },
        "readout_wf": {
            "type": "constant",
            "sample": 0.05,
        },
        "x180_wf": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.0027406153895280353), np.float64(0.006205924744757785), np.float64(0.010506399391870574), np.float64(0.01574250467874716), np.float64(0.021994570214268438), np.float64(0.029311470444420912), np.float64(0.03769896493843623), np.float64(0.047108819397196224), np.float64(0.057430016666353535), np.float64(0.06848342206009314), np.float64(0.08002114979608571), np.float64(0.09173156835624936), np.float64(0.10325039130369437), np.float64(0.11417766759589769), np.float64(0.12409978289982752), np.float64(0.13261490402078305), np.float64(0.13935974397909898), np.float64(0.1440351887320186)] + [0.14642827556545637] * 2 + [np.float64(0.1440351887320186), np.float64(0.13935974397909898), np.float64(0.13261490402078305), np.float64(0.12409978289982752), np.float64(0.11417766759589769), np.float64(0.10325039130369437), np.float64(0.09173156835624936), np.float64(0.08002114979608571), np.float64(0.06848342206009314), np.float64(0.057430016666353535), np.float64(0.047108819397196224), np.float64(0.03769896493843623), np.float64(0.029311470444420912), np.float64(0.021994570214268438), np.float64(0.01574250467874716), np.float64(0.010506399391870574), np.float64(0.006205924744757785), np.float64(0.0027406153895280353), np.float64(0.0)],
        },
        "x90_wf": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.0013703076947640177), np.float64(0.0031029623723788924), np.float64(0.005253199695935287), np.float64(0.00787125233937358), np.float64(0.010997285107134219), np.float64(0.014655735222210456), np.float64(0.018849482469218116), np.float64(0.023554409698598112), np.float64(0.028715008333176768), np.float64(0.03424171103004657), np.float64(0.040010574898042855), np.float64(0.04586578417812468), np.float64(0.051625195651847186), np.float64(0.057088833797948844), np.float64(0.06204989144991376), np.float64(0.06630745201039152), np.float64(0.06967987198954949), np.float64(0.0720175943660093)] + [0.07321413778272819] * 2 + [np.float64(0.0720175943660093), np.float64(0.06967987198954949), np.float64(0.06630745201039152), np.float64(0.06204989144991376), np.float64(0.057088833797948844), np.float64(0.051625195651847186), np.float64(0.04586578417812468), np.float64(0.040010574898042855), np.float64(0.03424171103004657), np.float64(0.028715008333176768), np.float64(0.023554409698598112), np.float64(0.018849482469218116), np.float64(0.014655735222210456), np.float64(0.010997285107134219), np.float64(0.00787125233937358), np.float64(0.005253199695935287), np.float64(0.0031029623723788924), np.float64(0.0013703076947640177), np.float64(0.0)],
        },
        "minus_x90_wf": {
            "type": "arbitrary",
            "samples": [np.float64(-0.0), np.float64(-0.0013703076947640177), np.float64(-0.0031029623723788924), np.float64(-0.005253199695935287), np.float64(-0.00787125233937358), np.float64(-0.010997285107134219), np.float64(-0.014655735222210456), np.float64(-0.018849482469218116), np.float64(-0.023554409698598112), np.float64(-0.028715008333176768), np.float64(-0.03424171103004657), np.float64(-0.040010574898042855), np.float64(-0.04586578417812468), np.float64(-0.051625195651847186), np.float64(-0.057088833797948844), np.float64(-0.06204989144991376), np.float64(-0.06630745201039152), np.float64(-0.06967987198954949), np.float64(-0.0720175943660093)] + [-0.07321413778272819] * 2 + [np.float64(-0.0720175943660093), np.float64(-0.06967987198954949), np.float64(-0.06630745201039152), np.float64(-0.06204989144991376), np.float64(-0.057088833797948844), np.float64(-0.051625195651847186), np.float64(-0.04586578417812468), np.float64(-0.040010574898042855), np.float64(-0.03424171103004657), np.float64(-0.028715008333176768), np.float64(-0.023554409698598112), np.float64(-0.018849482469218116), np.float64(-0.014655735222210456), np.float64(-0.010997285107134219), np.float64(-0.00787125233937358), np.float64(-0.005253199695935287), np.float64(-0.0031029623723788924), np.float64(-0.0013703076947640177), np.float64(-0.0)],
        },
    },
    "digital_waveforms": {
        "ON": {
            "samples": [(1, 0)],
        },
    },
    "integration_weights": {
        "cosine_weights": {
            "cosine": [(1.0, 1000)],
            "sine": [(0.0, 1000)],
        },
        "sine_weights": {
            "cosine": [(0.0, 1000)],
            "sine": [(1.0, 1000)],
        },
        "minus_sine_weights": {
            "cosine": [(0.0, 1000)],
            "sine": [(-1.0, 1000)],
        },
        "rotated_cosine_weights": {
            "cosine": [(np.float64(-0.40215491160240935), 1000)],
            "sine": [(np.float64(0.9155716395094697), 1000)],
        },
        "rotated_sine_weights": {
            "cosine": [(np.float64(-0.9155716395094697), 1000)],
            "sine": [(np.float64(-0.40215491160240935), 1000)],
        },
        "rotated_minus_sine_weights": {
            "cosine": [(np.float64(0.9155716395094697), 1000)],
            "sine": [(np.float64(0.40215491160240935), 1000)],
        },
    },
    "mixers": {
        "mixer_qubit": [{'intermediate_frequency': 123700073.93181705, 'lo_frequency': 5000000000.0, 'correction': (1, 0, 0, 1)}],
        "mixer_resonator": [{'intermediate_frequency': 90843700.88658619, 'lo_frequency': 6950000000.0, 'correction': (1, 0, 0, 1)}],
    },
}

loaded_config = {
    "controllers": {
        "con1": {
            "analog_inputs": {
                "1": {
                    "gain_db": 0,
                    "offset": 0.0011093127696236397,
                    "sampling_rate": 1000000000.0,
                    "shareable": False,
                },
                "2": {
                    "gain_db": 0,
                    "offset": -0.0009993545058356748,
                    "sampling_rate": 1000000000.0,
                    "shareable": False,
                },
            },
            "analog_outputs": {
                "1": {
                    "crosstalk": {},
                    "delay": 0,
                    "filter": {
                        "exponential": [],
                        "exponential_dc_gain": None,
                        "feedforward": [],
                        "high_pass": None,
                    },
                    "offset": 0.0,
                    "shareable": False,
                },
                "2": {
                    "crosstalk": {},
                    "delay": 0,
                    "filter": {
                        "exponential": [],
                        "exponential_dc_gain": None,
                        "feedforward": [],
                        "high_pass": None,
                    },
                    "offset": 0.0,
                    "shareable": False,
                },
                "3": {
                    "crosstalk": {},
                    "delay": 0,
                    "filter": {
                        "exponential": [],
                        "exponential_dc_gain": None,
                        "feedforward": [],
                        "high_pass": None,
                    },
                    "offset": 0.0,
                    "shareable": False,
                },
                "4": {
                    "crosstalk": {},
                    "delay": 0,
                    "filter": {
                        "exponential": [],
                        "exponential_dc_gain": None,
                        "feedforward": [],
                        "high_pass": None,
                    },
                    "offset": 0.0,
                    "shareable": False,
                },
            },
            "digital_inputs": {},
            "digital_outputs": {},
            "type": "opx1",
        },
    },
    "digital_waveforms": {
        "ON": {
            "samples": [(1, 0)],
        },
    },
    "elements": {
        "qubit": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "hold_offset": {
                "duration": 0,
            },
            "intermediate_frequency": 123700073.93181705,
            "mixInputs": {
                "I": ('con1', 1, 1),
                "Q": ('con1', 1, 2),
                "lo_frequency": 5000000000.0,
                "mixer": "mixer_qubit",
            },
            "operations": {
                "-x90": "-x90_pulse",
                "saturation": "saturation_pulse",
                "x180": "x180_pulse",
                "x90": "x90_pulse",
                "y90": "y90_pulse",
            },
            "outputs": {},
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
        },
        "resonator": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "hold_offset": {
                "duration": 0,
            },
            "intermediate_frequency": 90843700.88658619,
            "mixInputs": {
                "I": ('con1', 1, 3),
                "Q": ('con1', 1, 4),
                "lo_frequency": 6950000000.0,
                "mixer": "mixer_resonator",
            },
            "operations": {
                "readout": "readout_pulse",
            },
            "outputs": {
                "out1": ('con1', 1, 1),
                "out2": ('con1', 1, 2),
            },
            "smearing": 0,
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "time_of_flight": 236,
        },
    },
    "integration_weights": {
        "cosine_weights": {
            "cosine": [(1.0, 1000)],
            "sine": [(0.0, 1000)],
        },
        "minus_sine_weights": {
            "cosine": [(0.0, 1000)],
            "sine": [(-1.0, 1000)],
        },
        "rotated_cosine_weights": {
            "cosine": [(-0.40215491160240935, 1000)],
            "sine": [(0.9155716395094697, 1000)],
        },
        "rotated_minus_sine_weights": {
            "cosine": [(0.9155716395094697, 1000)],
            "sine": [(0.40215491160240935, 1000)],
        },
        "rotated_sine_weights": {
            "cosine": [(-0.9155716395094697, 1000)],
            "sine": [(-0.40215491160240935, 1000)],
        },
        "sine_weights": {
            "cosine": [(0.0, 1000)],
            "sine": [(1.0, 1000)],
        },
    },
    "mixers": {
        "mixer_qubit": [{'correction': (1.0, 0.0, 0.0, 1.0), 'intermediate_frequency': 123700073.93181705, 'lo_frequency': 5000000000.0}],
        "mixer_resonator": [{'correction': (1.0, 0.0, 0.0, 1.0), 'intermediate_frequency': 90843700.88658619, 'lo_frequency': 6950000000.0}],
    },
    "oscillators": {},
    "pulses": {
        "-x90_pulse": {
            "integration_weights": {},
            "length": 40,
            "operation": "control",
            "waveforms": {
                "I": "minus_x90_wf",
                "Q": "zero_wf",
            },
        },
        "readout_pulse": {
            "digital_marker": "ON",
            "integration_weights": {
                "cos": "cosine_weights",
                "minus_sin": "minus_sine_weights",
                "rotated_cos": "rotated_cosine_weights",
                "rotated_minus_sin": "rotated_minus_sine_weights",
                "rotated_sin": "rotated_sine_weights",
                "sin": "sine_weights",
            },
            "length": 1000,
            "operation": "measurement",
            "waveforms": {
                "I": "readout_wf",
                "Q": "zero_wf",
            },
        },
        "saturation_pulse": {
            "integration_weights": {},
            "length": 20000,
            "operation": "control",
            "waveforms": {
                "I": "saturation_wf",
                "Q": "zero_wf",
            },
        },
        "x180_pulse": {
            "integration_weights": {},
            "length": 40,
            "operation": "control",
            "waveforms": {
                "I": "x180_wf",
                "Q": "zero_wf",
            },
        },
        "x90_pulse": {
            "integration_weights": {},
            "length": 40,
            "operation": "control",
            "waveforms": {
                "I": "x90_wf",
                "Q": "zero_wf",
            },
        },
        "y90_pulse": {
            "integration_weights": {},
            "length": 40,
            "operation": "control",
            "waveforms": {
                "I": "zero_wf",
                "Q": "x90_wf",
            },
        },
    },
    "waveforms": {
        "minus_x90_wf": {
            "is_overridable": False,
            "max_allowed_error": 0.0001,
            "samples": [-0.0, -0.0013703076947640177, -0.0031029623723788924, -0.005253199695935287, -0.00787125233937358, -0.010997285107134219, -0.014655735222210456, -0.018849482469218116, -0.023554409698598112, -0.028715008333176768, -0.03424171103004657, -0.040010574898042855, -0.04586578417812468, -0.051625195651847186, -0.057088833797948844, -0.06204989144991376, -0.06630745201039152, -0.06967987198954949, -0.0720175943660093] + [-0.07321413778272819] * 2 + [-0.0720175943660093, -0.06967987198954949, -0.06630745201039152, -0.06204989144991376, -0.057088833797948844, -0.051625195651847186, -0.04586578417812468, -0.040010574898042855, -0.03424171103004657, -0.028715008333176768, -0.023554409698598112, -0.018849482469218116, -0.014655735222210456, -0.010997285107134219, -0.00787125233937358, -0.005253199695935287, -0.0031029623723788924, -0.0013703076947640177, -0.0],
            "type": "arbitrary",
        },
        "readout_wf": {
            "sample": 0.05,
            "type": "constant",
        },
        "saturation_wf": {
            "sample": 0.002,
            "type": "constant",
        },
        "x180_wf": {
            "is_overridable": False,
            "max_allowed_error": 0.0001,
            "samples": [0.0, 0.0027406153895280353, 0.006205924744757785, 0.010506399391870574, 0.01574250467874716, 0.021994570214268438, 0.029311470444420912, 0.03769896493843623, 0.047108819397196224, 0.057430016666353535, 0.06848342206009314, 0.08002114979608571, 0.09173156835624936, 0.10325039130369437, 0.11417766759589769, 0.12409978289982752, 0.13261490402078305, 0.13935974397909898, 0.1440351887320186] + [0.14642827556545637] * 2 + [0.1440351887320186, 0.13935974397909898, 0.13261490402078305, 0.12409978289982752, 0.11417766759589769, 0.10325039130369437, 0.09173156835624936, 0.08002114979608571, 0.06848342206009314, 0.057430016666353535, 0.047108819397196224, 0.03769896493843623, 0.029311470444420912, 0.021994570214268438, 0.01574250467874716, 0.010506399391870574, 0.006205924744757785, 0.0027406153895280353, 0.0],
            "type": "arbitrary",
        },
        "x90_wf": {
            "is_overridable": False,
            "max_allowed_error": 0.0001,
            "samples": [0.0, 0.0013703076947640177, 0.0031029623723788924, 0.005253199695935287, 0.00787125233937358, 0.010997285107134219, 0.014655735222210456, 0.018849482469218116, 0.023554409698598112, 0.028715008333176768, 0.03424171103004657, 0.040010574898042855, 0.04586578417812468, 0.051625195651847186, 0.057088833797948844, 0.06204989144991376, 0.06630745201039152, 0.06967987198954949, 0.0720175943660093] + [0.07321413778272819] * 2 + [0.0720175943660093, 0.06967987198954949, 0.06630745201039152, 0.06204989144991376, 0.057088833797948844, 0.051625195651847186, 0.04586578417812468, 0.040010574898042855, 0.03424171103004657, 0.028715008333176768, 0.023554409698598112, 0.018849482469218116, 0.014655735222210456, 0.010997285107134219, 0.00787125233937358, 0.005253199695935287, 0.0031029623723788924, 0.0013703076947640177, 0.0],
            "type": "arbitrary",
        },
        "zero_wf": {
            "sample": 0.0,
            "type": "constant",
        },
    },
}

