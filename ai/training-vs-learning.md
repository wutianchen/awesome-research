# Training vs learning

**Learning** is the outcome: the model gets better at a task. **Training** is the
procedure that makes it happen: running an optimiser on a loss over data.

> "A computer program is said to **learn** from experience *E* with respect to some class
> of tasks *T* and performance measure *P*, if its performance at tasks in *T*, as
> measured by *P*, improves with experience *E*."
> — Tom Mitchell, *Machine Learning* (1997)

| | Training | Learning |
|---|---|---|
| What it is | An engineering process: an optimiser (gradient descent) on a loss over a dataset | A capability: improving at a task from experience |
| Measured by | Training loss going down | Performance on **unseen** data (generalisation) |
| Can happen without the other | **Overfitting**: training loss falls, the model memorises, nothing general is learned | **In-context learning**: weights are frozen, the model picks up a task from the prompt |

That's why the words combine as they do. **Learning paradigms** (self-supervised,
supervised, reinforcement) say *where the learning signal comes from*. **Training stages**
(pre-, mid-, post-) say *when, and on what data, the optimiser runs*.