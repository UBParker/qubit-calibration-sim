
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
    with for_(v1,0,(v1<100),(v1+1)):
        save(v1, r1)
        with for_(v2,80000000,(v2<100000000),(v2+50000)):
            update_frequency('resonator', v2, 'Hz', False)
            measure('readout', 'resonator', dual_demod.full("cos", "sin", v3), dual_demod.full("minus_sin", "cos", v4))
            wait(250, 'resonator')
            save(v3, r2)
            save(v4, r3)
    with stream_processing():
        r1.save("n")
        r2.buffer(400).average().save("I")
        r3.buffer(400).average().save("Q")

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
            "intermediate_frequency": 120000000.0,
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
            "intermediate_frequency": 90000000.0,
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
            "samples": [np.float64(0.0), np.float64(0.0018677942290391875), np.float64(0.004229484541465065), np.float64(0.0071603597597458294), np.float64(0.010728889395403278), np.float64(0.014989819977433047), np.float64(0.019976460597110737), np.float64(0.025692736537135163), np.float64(0.03210577498146693), np.float64(0.039139915112828785), np.float64(0.04667307240463428), np.float64(0.05453630683141879), np.float64(0.06251723414062206), np.float64(0.07036758450673712), np.float64(0.0778147818317178), np.float64(0.08457693816177221), np.float64(0.09038019466761463), np.float64(0.09497696267748301), np.float64(0.09816338889440603)] + [0.09979433419018298] * 2 + [np.float64(0.09816338889440603), np.float64(0.09497696267748301), np.float64(0.09038019466761463), np.float64(0.08457693816177221), np.float64(0.0778147818317178), np.float64(0.07036758450673712), np.float64(0.06251723414062206), np.float64(0.05453630683141879), np.float64(0.04667307240463428), np.float64(0.039139915112828785), np.float64(0.03210577498146693), np.float64(0.025692736537135163), np.float64(0.019976460597110737), np.float64(0.014989819977433047), np.float64(0.010728889395403278), np.float64(0.0071603597597458294), np.float64(0.004229484541465065), np.float64(0.0018677942290391875), np.float64(0.0)],
        },
        "x90_wf": {
            "type": "arbitrary",
            "samples": [np.float64(0.0), np.float64(0.0009338971145195937), np.float64(0.0021147422707325325), np.float64(0.0035801798798729147), np.float64(0.005364444697701639), np.float64(0.007494909988716524), np.float64(0.009988230298555369), np.float64(0.012846368268567582), np.float64(0.016052887490733466), np.float64(0.019569957556414393), np.float64(0.02333653620231714), np.float64(0.027268153415709394), np.float64(0.03125861707031103), np.float64(0.03518379225336856), np.float64(0.0389073909158589), np.float64(0.042288469080886106), np.float64(0.045190097333807314), np.float64(0.047488481338741506), np.float64(0.04908169444720301)] + [0.04989716709509149] * 2 + [np.float64(0.04908169444720301), np.float64(0.047488481338741506), np.float64(0.045190097333807314), np.float64(0.042288469080886106), np.float64(0.0389073909158589), np.float64(0.03518379225336856), np.float64(0.03125861707031103), np.float64(0.027268153415709394), np.float64(0.02333653620231714), np.float64(0.019569957556414393), np.float64(0.016052887490733466), np.float64(0.012846368268567582), np.float64(0.009988230298555369), np.float64(0.007494909988716524), np.float64(0.005364444697701639), np.float64(0.0035801798798729147), np.float64(0.0021147422707325325), np.float64(0.0009338971145195937), np.float64(0.0)],
        },
        "minus_x90_wf": {
            "type": "arbitrary",
            "samples": [np.float64(-0.0), np.float64(-0.0009338971145195937), np.float64(-0.0021147422707325325), np.float64(-0.0035801798798729147), np.float64(-0.005364444697701639), np.float64(-0.007494909988716524), np.float64(-0.009988230298555369), np.float64(-0.012846368268567582), np.float64(-0.016052887490733466), np.float64(-0.019569957556414393), np.float64(-0.02333653620231714), np.float64(-0.027268153415709394), np.float64(-0.03125861707031103), np.float64(-0.03518379225336856), np.float64(-0.0389073909158589), np.float64(-0.042288469080886106), np.float64(-0.045190097333807314), np.float64(-0.047488481338741506), np.float64(-0.04908169444720301)] + [-0.04989716709509149] * 2 + [np.float64(-0.04908169444720301), np.float64(-0.047488481338741506), np.float64(-0.045190097333807314), np.float64(-0.042288469080886106), np.float64(-0.0389073909158589), np.float64(-0.03518379225336856), np.float64(-0.03125861707031103), np.float64(-0.027268153415709394), np.float64(-0.02333653620231714), np.float64(-0.019569957556414393), np.float64(-0.016052887490733466), np.float64(-0.012846368268567582), np.float64(-0.009988230298555369), np.float64(-0.007494909988716524), np.float64(-0.005364444697701639), np.float64(-0.0035801798798729147), np.float64(-0.0021147422707325325), np.float64(-0.0009338971145195937), np.float64(-0.0)],
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
            "cosine": [(np.float64(1.0), 1000)],
            "sine": [(np.float64(-0.0), 1000)],
        },
        "rotated_sine_weights": {
            "cosine": [(np.float64(0.0), 1000)],
            "sine": [(np.float64(1.0), 1000)],
        },
        "rotated_minus_sine_weights": {
            "cosine": [(np.float64(-0.0), 1000)],
            "sine": [(np.float64(-1.0), 1000)],
        },
    },
    "mixers": {
        "mixer_qubit": [{'intermediate_frequency': 120000000.0, 'lo_frequency': 5000000000.0, 'correction': (1, 0, 0, 1)}],
        "mixer_resonator": [{'intermediate_frequency': 90000000.0, 'lo_frequency': 6950000000.0, 'correction': (1, 0, 0, 1)}],
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
            "intermediate_frequency": 120000000.0,
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
            "intermediate_frequency": 90000000.0,
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
            "cosine": [(1.0, 1000)],
            "sine": [(-0.0, 1000)],
        },
        "rotated_minus_sine_weights": {
            "cosine": [(-0.0, 1000)],
            "sine": [(-1.0, 1000)],
        },
        "rotated_sine_weights": {
            "cosine": [(0.0, 1000)],
            "sine": [(1.0, 1000)],
        },
        "sine_weights": {
            "cosine": [(0.0, 1000)],
            "sine": [(1.0, 1000)],
        },
    },
    "mixers": {
        "mixer_qubit": [{'correction': (1.0, 0.0, 0.0, 1.0), 'intermediate_frequency': 120000000.0, 'lo_frequency': 5000000000.0}],
        "mixer_resonator": [{'correction': (1.0, 0.0, 0.0, 1.0), 'intermediate_frequency': 90000000.0, 'lo_frequency': 6950000000.0}],
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
            "samples": [-0.0, -0.0009338971145195937, -0.0021147422707325325, -0.0035801798798729147, -0.005364444697701639, -0.007494909988716524, -0.009988230298555369, -0.012846368268567582, -0.016052887490733466, -0.019569957556414393, -0.02333653620231714, -0.027268153415709394, -0.03125861707031103, -0.03518379225336856, -0.0389073909158589, -0.042288469080886106, -0.045190097333807314, -0.047488481338741506, -0.04908169444720301] + [-0.04989716709509149] * 2 + [-0.04908169444720301, -0.047488481338741506, -0.045190097333807314, -0.042288469080886106, -0.0389073909158589, -0.03518379225336856, -0.03125861707031103, -0.027268153415709394, -0.02333653620231714, -0.019569957556414393, -0.016052887490733466, -0.012846368268567582, -0.009988230298555369, -0.007494909988716524, -0.005364444697701639, -0.0035801798798729147, -0.0021147422707325325, -0.0009338971145195937, -0.0],
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
            "samples": [0.0, 0.0018677942290391875, 0.004229484541465065, 0.0071603597597458294, 0.010728889395403278, 0.014989819977433047, 0.019976460597110737, 0.025692736537135163, 0.03210577498146693, 0.039139915112828785, 0.04667307240463428, 0.05453630683141879, 0.06251723414062206, 0.07036758450673712, 0.0778147818317178, 0.08457693816177221, 0.09038019466761463, 0.09497696267748301, 0.09816338889440603] + [0.09979433419018298] * 2 + [0.09816338889440603, 0.09497696267748301, 0.09038019466761463, 0.08457693816177221, 0.0778147818317178, 0.07036758450673712, 0.06251723414062206, 0.05453630683141879, 0.04667307240463428, 0.039139915112828785, 0.03210577498146693, 0.025692736537135163, 0.019976460597110737, 0.014989819977433047, 0.010728889395403278, 0.0071603597597458294, 0.004229484541465065, 0.0018677942290391875, 0.0],
            "type": "arbitrary",
        },
        "x90_wf": {
            "is_overridable": False,
            "max_allowed_error": 0.0001,
            "samples": [0.0, 0.0009338971145195937, 0.0021147422707325325, 0.0035801798798729147, 0.005364444697701639, 0.007494909988716524, 0.009988230298555369, 0.012846368268567582, 0.016052887490733466, 0.019569957556414393, 0.02333653620231714, 0.027268153415709394, 0.03125861707031103, 0.03518379225336856, 0.0389073909158589, 0.042288469080886106, 0.045190097333807314, 0.047488481338741506, 0.04908169444720301] + [0.04989716709509149] * 2 + [0.04908169444720301, 0.047488481338741506, 0.045190097333807314, 0.042288469080886106, 0.0389073909158589, 0.03518379225336856, 0.03125861707031103, 0.027268153415709394, 0.02333653620231714, 0.019569957556414393, 0.016052887490733466, 0.012846368268567582, 0.009988230298555369, 0.007494909988716524, 0.005364444697701639, 0.0035801798798729147, 0.0021147422707325325, 0.0009338971145195937, 0.0],
            "type": "arbitrary",
        },
        "zero_wf": {
            "sample": 0.0,
            "type": "constant",
        },
    },
}

