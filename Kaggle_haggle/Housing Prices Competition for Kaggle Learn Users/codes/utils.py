import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def x_y_scatterplots(data, target, ncols=4, cols=[]):
    train_df = data

    # Only select specific columns if provided
    if len(cols) > 0:
        if target not in cols:
            cols = pd.concat([cols, pd.Series(target)])
        train_df = train_df[cols]
    
    # Create scatter plots for all features against target
    fig, axes = plt.subplots(
        nrows=int(np.ceil(len(train_df.columns) / ncols)),
        ncols=ncols,
        figsize=(15, 3 * int(np.ceil(len(train_df.columns) / ncols))),
        layout="constrained"
    )
    
    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    
    for i, col in enumerate(train_df[train_df.columns.drop(target)]):
        # print(f"Plotting {i}. {col} ...")
        not_nan_idx = train_df[col].dropna().index
        dropped_na_count = train_df[col].isna().sum()

        axes[i].scatter(
            train_df[col][not_nan_idx], train_df[target][not_nan_idx],
            edgecolors='blue', alpha=0.5
        )
        # axes[i].set_xlabel(col)
        # axes[i].set_ylabel("Y")
        axes[i].set_title(f"{col}")
        axes[i].tick_params(axis="x", labelrotation=45)
        
        # Superimpose NaNs Counts if NaNs were dropped
        if dropped_na_count > 0:
            axes[i].text(
                0.95, 0.95,
                f'Dropped NA: {dropped_na_count}', 
                verticalalignment='top', horizontalalignment='right',
                transform=axes[i].transAxes,
                fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='black')
            )
                 
    # Delete any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    