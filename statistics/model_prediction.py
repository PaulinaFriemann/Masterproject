from typing import NamedTuple
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()


def plot(data, x, y):
    
    print(data)
    sns.relplot(data=data, x=x, y=y)
    plt.show()


def end_probability_tm(data, tm, all_models):
    p_tm = data[tm].iloc[-1]

    all_probs = data.loc[:, all_models].iloc[[-1]]
    #sel = all_probs.loc[all_probs.index.map(lambda x: tm.solver in x)]
    #sel = sel * sum(sel)
    all_probs = pd.melt(all_probs)
    all_probs.columns = ['Representation', 'Strategy', 'Determinism', 'P']
    all_probs["Determinism"] = all_probs["Determinism"].apply(pd.to_numeric)
    print(all_probs.dtypes)

    sns.catplot(x="Determinism", row="Strategy", col="Representation", data=all_probs)
    #fg = sns.factorplot(x=)
    #print(p_tm)
    plt.show()
    exit()


class Model(NamedTuple):
    rep: str
    solver: str
    det: float


df = pd.read_csv("results/model_prediction_1.csv")
print(df)
print()


def f(lis):
    arr = [lis[0]]
    for x in lis:
        if x != arr[-1]:
            arr.append(x)
    return arr


print(f(df["run"]))

exit()
models = df.iloc[:, df.columns.get_loc("t") + 1:].columns.str.split('_')
models = [tuple(m[1:]) for m in models]
models = pd.MultiIndex.from_tuples(models)
cols = pd.MultiIndex.from_tuples([(c, '', '') for c in df.columns[ : df.columns.get_loc("t") + 1]]).append(models)
df.columns = cols

runs = df['run'].max()

for n in range(runs + 1):
    run = df[df['run'] == n]
    print(run)

    true_model = run['true_model'][0]
    true_model_obj = Model(*(true_model).split('_'))
    print(true_model_obj)
    end_probability_tm(run, true_model_obj, models)
    exit()
    plot(run, "t", "p_{}".format(true_model)) #["p_{}".format(true_model)])
    exit()
