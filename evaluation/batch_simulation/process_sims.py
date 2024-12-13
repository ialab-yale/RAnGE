import pickle

import numpy as np

if __name__ == "__main__":
    for noise in ["rau", "ran", "wc", "none"]:
        Es = []
        Js = []
        for i in range(3):
            with open(
                f"RAnGE/logs/WAFR_bimodal_{i}/plots/results_{noise}.pkl", "rb"
            ) as f:
                res = pickle.load(f)
                Es.append(res["Es"])
                Js.append(res["Js"])

        print("--------\n----------")
        print(f"Noise: {noise}")
        print(f"E")
        print(f"Min: {np.min(Es)}")
        print(f"Max: {np.max(Es)}")
        print(f"Mean: {np.average(Es)}")
        print(f"Std: {np.std(Es)}")
        print("----------------------")
        print(f"J")
        print(f"Min: {np.min(Js)}")
        print(f"Max: {np.max(Js)}")
        print(f"Mean: {np.average(Js)}")
        print(f"Std: {np.std(Js)}")
