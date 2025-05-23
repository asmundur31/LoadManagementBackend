import pandas as pd
import numpy as np
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd


### Functions for Reliability Analysis ###
def prepare_icc_data(data, condition_name):
    df = pd.DataFrame(data)
    df_long = df.reset_index().melt(id_vars='index', var_name='rater', value_name='score')
    df_long = df_long.rename(columns={'index': 'subject'})
    df_long['condition'] = condition_name
    # Add a mean score for the missing values
    df_long['score'] = df_long.groupby('subject')['score'].transform(lambda x: x.fillna(x.mean()))
    return df_long

def calculate_icc(data, condition_name, model='ICC3'):
    df_long = prepare_icc_data(data, condition_name)
    icc = pg.intraclass_corr(data=df_long, targets='subject', raters='rater', ratings='score')
    return icc[icc['Type'] == model]

def perform_t_test(data1, data2):
    # Perform paired t-test
    data1 = np.array(data1)
    data2 = np.array(data2)

    # Mean per subject
    data1_means = np.nanmean(data1, axis=1)
    data2_means = np.nanmean(data2, axis=1)

    t_test_result = pg.ttest(data2_means, data1_means, paired=True)
    return t_test_result

def perform_anova(comp, semi, soft):
    # Create a DataFrame for each condition, stacking the data into long format
    comp_df = pd.DataFrame(comp, columns=[f'Measure_{i+1}' for i in range(4)])
    comp_df['condition'] = 'competition'
    comp_df['participant'] = np.arange(1, 10)

    semi_df = pd.DataFrame(semi, columns=[f'Measure_{i+1}' for i in range(4)])
    semi_df['condition'] = 'semi-soft'
    semi_df['participant'] = np.arange(1, 10)

    soft_df = pd.DataFrame(soft, columns=[f'Measure_{i+1}' for i in range(4)])
    soft_df['condition'] = 'soft'
    soft_df['participant'] = np.arange(1, 10)

    # Combine all data into one DataFrame
    df = pd.concat([comp_df, semi_df, soft_df], ignore_index=True)

    # Fill NaNs with the participant's row mean (across their 4 measures only)
    measure_cols = [f'Measure_{i+1}' for i in range(4)]
    df[measure_cols] = df[measure_cols].T.apply(lambda x: x.fillna(x.mean()), axis=0).T

    # Reshape the data into long format
    df_long = pd.melt(df, id_vars=['participant', 'condition'], value_vars=[f'Measure_{i+1}' for i in range(4)],
                    var_name='measure', value_name='jump_height')

    # Perform repeated measures ANOVA
    anova_result = pg.rm_anova(dv='jump_height', within='condition', subject='participant', data=df_long, detailed=True)

    return anova_result

def perform_tukey_test(comp, semi, soft, feature_name="feature"):
    # Convert input arrays to DataFrames (subjects as rows, measures as columns)
    comp_df = pd.DataFrame(comp)
    semi_df = pd.DataFrame(semi)
    soft_df = pd.DataFrame(soft)
    

    # Fill NaNs with the subject (row) mean
    comp_df = comp_df.T.fillna(comp_df.T.mean()).T
    semi_df = semi_df.T.fillna(semi_df.T.mean()).T
    soft_df = soft_df.T.fillna(soft_df.T.mean()).T

    # Flatten all arrays
    comp_flat = comp_df.values.flatten()
    semi_flat = semi_df.values.flatten()
    soft_flat = soft_df.values.flatten()

    # Create corresponding group labels
    comp_labels = ['comp'] * len(comp_flat)
    semi_labels = ['semi'] * len(semi_flat)
    soft_labels = ['soft'] * len(soft_flat)

    # Combine into one DataFrame
    all_values = np.concatenate([soft_flat, semi_flat, comp_flat])
    all_labels = soft_labels + semi_labels + comp_labels

    df = pd.DataFrame({
        feature_name: all_values,
        'landing_type': all_labels
    })

    # Run Tukey's HSD
    tukey = pairwise_tukeyhsd(endog=df[feature_name], groups=df['landing_type'], alpha=0.05)
    return tukey.summary()

### Functions for Validity Analysis ###
def pearsons_correlation(data1, data2):
    # Calculate Pearson's correlation coefficient
    data1 = np.array(data1)
    data2 = np.array(data2)

    # Mean per subject
    data1_means = np.nanmean(data1, axis=1)
    data2_means = np.nanmean(data2, axis=1)

    correlation = np.corrcoef(data1_means, data2_means)[0, 1]
    return correlation