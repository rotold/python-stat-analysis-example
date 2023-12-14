""" In order not to replicate the work of others who published
results on the dataset in Kaggle, their work generally will not be replicated"""

import pandas as pd
import scipy.stats as stats
from matplotlib.pyplot import show
import plotly.express as px


def clean_data(new_path, path=None):
    """ Clean the datafile having path path - EDA was already done by others, which show that the only
    data that need to be cleaned is to replace the null data in the 'Returns' column with the number zero.
    If necessary a cleaned csv with path new_path for subsequent analysis will be created.
    and return a corresponding cleaned DataFrame.
    Some other checks to verify assumptions made by the analysis are also performed.

    :param new_path: the path in which to put the csv representation of cleaned data if the path parameter is not
    None
    :param path: the path of the csv DataFrame representation Kaggle data under consideration.
    :return: the cleaned DataFrame
    """

    # remove the limit on the number of columns displayed in a DataFrame
    pd.set_option('display.max_columns', None)

    # if cleaned csv is already made, create a DataFrame therefrom and return it
    if path is None:
        df = pd.read_csv(new_path, index_col='Record')
        return df
    # Otherwise we must clean and check the original version
    print("Cleaning and checking data")
    df = pd.read_csv(path)

    # make sure results are the same as reported by other publishers; Only 'Returns'
    # should show
    bool_series = df.isnull().sum() > 0
    print(f"This must show only 'Returns' if 'Returns' is the only columns with nulls:"
          f" {bool_series[bool_series].index.format()}")
    # Assume a null return means zero returns and replace nulls with zero
    df.fillna(0, inplace=True)

    # see if 'Customer ID' and 'Purchase Date' are a valid "database key":
    x = df.groupby(['Customer ID', 'Purchase Date']).count().max() > 1
    print(f"If a unique 'Customer ID' and 'Purchase Date' determine a unique row\n"
          f"then this should be 0: {x[x].size}")

    # Give a name 'Record' to the index of df
    df.index.names = ['Record']
    # create the new csv and return the cleaned DataFrame
    df.to_csv(new_path)
    return df


def tableau_verify(df: pd.DataFrame):
    """Create a normalized cumulative sum for the cumulative percent of individuals vs Age and Gender.
    This is used to verify the cumulative distribution data done in Tableau.

    :param df: the DataFrame of the Kaggle data under consideration
    """

    x = df[['Customer ID', 'Age', 'Gender']].groupby(['Customer ID'], as_index=False).max()
    crstb = pd.crosstab(x['Age'], columns=x['Gender'])
    crstb.sort_values(by='Age')
    num_males, num_females = crstb['Male'].sum(), crstb['Female'].sum()
    # Get the Cum Sum as a percentage as was done in Tableau
    print(f"Number of Males: {num_males} Females: {num_females}")
    crstb[['Male', 'Female']] = crstb[['Male', 'Female']].cumsum(0)
    crstb['Male'] = crstb['Male']/num_males*100
    crstb['Female'] = crstb['Female']/num_females*100

    # see if matches Tableau at point 30 years and Male
    print(f"Python: {crstb.at[30, 'Male']} Tableau: 25.12")
    # output file for reference
    crstb.to_csv("Cum percent Age Gender Tableau")


def create_individ_stats(df: pd.DataFrame) -> pd.DataFrame:
    """This creates statistics of each individual in the data grouping by 'Customer ID'.
    It takes into account that each data point (row) in the source data is of a particular purchase of an individual.
    It then returns a DataFrame of the stats for further analysis.

    :param: df: The DataFrame of the Kaggle Data under consideration.
    :return: A new DataFrame with each row representing the statistics of the individual
    """

    def pay_meth(x: pd.Series):
        """A nested function that categorizes each individual as solely a credit buyer, a cash buyer or a mixture,
        and returns an appropriate string"""
        if ('PayPal' in x.values or 'Credit Card' in x.values) and 'Cash' in x.values:
            return 'Mixed'
        if 'Cash' not in x.values:
            return 'Non-cash'
        else:
            return 'Cash Only'

    def cat_info(x: pd.DataFrame):
        """A nested function that calculates the percent of total money a customer spent for a category as percentage
        of the total the customer spend among all the purchases. It returns a DataFrame of the data."""
        # create stats variables
        books, clothing, home, electronics, total, error, num_pur = 0, 0, 0, 0, 0, 0, 0
        gndr, age, cust_id = None, None, None

        for cat_data in x.itertuples(index=False, name=None):
            cust_id, category, gndr, age, cost = cat_data
            # print(cat_data)
            # cal statistic based on percent of total purchases - assume time of purchase not
            # a confounding factor in real life situation
            # total verified to be non-zero

            total = total + cost
            num_pur = num_pur + 1

            if category == 'Books':
                books = books + cost
            elif category == 'Clothing':
                clothing = clothing + cost
            elif category == 'Home':
                home = home + cost
            elif category == 'Electronics':
                electronics = electronics + cost
            else:
                # it was known beforehand that an error would not occur
                error = error + cost
                print("!______ERROR_______")

        # division by zero is not an issue for the dataset.
        data = {'Gender': gndr, 'Age': age, 'Book pct': books / total * 100, 'Clothes pct': clothing / total * 100,
                'Home pct':
                    home / total * 100, 'Electron pct': electronics / total * 100, 'Num Purchases': num_pur}

        # print(f"b: {books} c: {clothing} h: {home} e: {electronics} t: {total} e: {error}")
        # dummy value till done
        ind = [cust_id]
        out = pd.DataFrame(data, index=ind)
        out.index.names = ['custid']
        return out

    by_individuals = (df[['Customer ID', 'Age', 'Product Category', 'Quantity', 'Payment Method',
                          'Total Purchase Amount', 'Returns', 'Gender', 'Churn']]
                      .groupby('Customer ID'))
    pay_meth_data = by_individuals['Payment Method'].apply(pay_meth)

    cat_rates_data = (by_individuals[['Customer ID', 'Product Category', 'Gender', 'Age', 'Total Purchase Amount']]
                      .apply(cat_info))
    # drop the extra index created by apply(cat_info)
    cat_rates_data.reset_index(level='custid', drop=True, inplace=True)

    # Note: using this way is MUCH faster than using apply and a callable!
    churn_return_data = (by_individuals[['Returns', 'Churn']].sum() /
                         by_individuals[['Returns', 'Churn']].count()*100)
    churn_return_data.rename(columns={'Returns': 'Return Rate', 'Churn': 'Churned?'}, inplace=True)
    # In the orginal data the churned flag is either true or false for all rows of a Customer ID
    churn_return_data['Churned?'] = churn_return_data['Churned?'].map({0: 'No', 100: 'Yes'})

    join_tables = [pay_meth_data, churn_return_data]
    all_indiv_data = cat_rates_data.join(join_tables)

    return all_indiv_data


def conting_table(x: pd.DataFrame):
    """See whether the payment method and churn rate of customers is independent of gender using
    Chi-square test on a contingency table.

    :param x: The DatFrame of the individual data created by create_individ_stats
    """
    # Look at purchase method versus Gender
    ct = pd.crosstab(x['Gender'], [x['Payment Method']], margins=True)
    print(f"Contingency table with margin totals:\n{ct}")
    ct.to_csv("conting table")
    ct = pd.crosstab(x['Gender'], [x['Payment Method']], margins=False)
    chi2, pvalue, dof, expect_freq = stats.chi2_contingency(ct)
    print(f"Chi^2 {chi2} with {dof} degrees of freedom has a p-value of {pvalue}")
    print("\nTherefore we can not reject the hypothesis that, when given the choice of PayPal"
          "\n Credit Card, and Cash, the customer propensity for using"
          "\n only cash, only non-cash, or a mixture of both does not depend on gender.")

    pd.DataFrame(expect_freq).to_csv("expected freq")

    print("\n\nWe now look at the Curn Rate Contingency Table:")
    ct = pd.crosstab(x['Gender'], columns=x['Churned?'], margins=True)
    ct.to_csv("curn conting table")
    print(ct)
    ct = pd.crosstab(x['Gender'], columns=x['Churned?'], margins=False)
    chi2, pvalue, dof, expect_freq = stats.chi2_contingency(ct)
    print(f"\nChi^2 {chi2} with {dof} degrees of freedom has a p-value of {pvalue}")
    print(f"Therefore we do not reject the hypothesis that Churn rate does not depend on Gender")
    print(f"\nAccordingly the expected values are very close to the actual values for this high p-value result:")
    print(expect_freq)
    pd.DataFrame(expect_freq).to_csv("Expected Churn Freq")


def rate_tests(df: pd.DataFrame):
    """
    A function that creates histograms of the percentages that individuals spend on the various
    product categories, as well as the return rate of customers (percentage of purchases that were returned),
    and determines whether the rates are independent of gender for data which is filtered from the extreme
    values of 0% and 100%

    :param df: The DatFrame of the individual data created by create_individ_stats
    """

    # Function to make cumulative distributions of rates
    rates = ['Book pct', 'Clothes pct', 'Home pct', 'Electron pct', 'Return Rate']
    titles = ['Percent of Purchases Spent on Books', 'Percent of Purchases Spent on Clothes',
              'Percent of Purchases Spent on Home Goods', 'Percent of Purchases Spent on Electronics',
              'Percent of Orders That Were Returned']

    def wilcoxon_rank_sum(wdf: pd.DataFrame):
        """
        Uses Wilcoxon rank-sum test to determine if the rate data which have been filtered of their 0% and 100%
        values is independent of gender

        :param wdf: The DatFrame of the individual data created by create_individ_stats
        :return:
        """

        print("We now get an idea if the rates are dependent on gender on these skewed distributions that\n"
              "were clipped of their 0 and 100% values by looking at the Wilcoxon Rank-Sum Test.\n"
              "We note the similarity in shape between the male and female populations for a given\n"
              "rate category.\n")

        print("Let us now look at the percentage of ties in each category:")
        for i in rates:
            print(f"% of unique values in {i}: {wdf[i].nunique(dropna=True)/wdf[i].count() * 100}")

        print("We see that for every category except the Return Rate category, less than 0.1% of the data\n"
              "are duplicates.  As expected, virtually all the data of the return rates have duplicates.\n"
              "Thus for the non-return-rate categories we shall accept the approximation of deleting the\n"
              "duplicate values of those rates. For the Return Rate, we shall use the scipy.stats.mannwhitneyu\n"
              "method which corrects for ties by approximating with the Normal Distribution.")

        # make a DataFrame for the stats:
        rate_stats = pd.DataFrame(index=rates, columns=['Stat_Value', 'p-value', 'male median', 'female median'])
        for i in rates[:4]:
            # skip return rate
            # create a Series with no duplicates
            no_dups = wdf[['Gender', i]].drop_duplicates(subset=i, keep=False, inplace=False, ignore_index=False)
            m_in = no_dups[no_dups['Gender'] == 'Male']
            f_in = no_dups[no_dups['Gender'] == 'Female']
            del no_dups

            # get stats
            w_stats = stats.ranksums(m_in[i], f_in[i], 'two-sided', nan_policy='omit')
            m_median = m_in[i].median()
            f_median = f_in[i].median()
            rate_stats.loc[i] = [w_stats[0], w_stats[1], m_median, f_median]

        print(f"Here are the statistical results:\n{rate_stats}")
        rate_stats.to_csv("rate_stats_2sided")
        print("We see that the medians are very close for all categories between male and female.\n"
              "We also see that the p-values are such that we do not reject the hypothesis that the male and female\n"
              "populations are the same.  The statistic appears to be the standard normal.")

        print("If we do one sided tests, we would find that the smallest p-value occurs in the one sided test having\n"
              "the alternative hypothesis of the Electronics Category rate of males being less than \n"
              "that of the females. The p-value was 0.288. As expected, this is not a significant p-value.\n")

        print("We now look at the return rate which has many ties")
        ret_data = wdf[['Gender', 'Return Rate']]
        males = ret_data[ret_data['Gender'] == 'Male']
        females = ret_data[ret_data['Gender'] == 'Female']
        rm_med = males['Return Rate'].median()
        rf_med = females['Return Rate'].median()
        # get the statistics for this test
        q = stats.mannwhitneyu(males['Return Rate'], females['Return Rate'], alternative='two-sided',
                               method='asymptotic', nan_policy='omit')
        print(f"The statistic value is {q[0]} and the p-value is {q[1]} and the median value for males\n"
              f"is: {rm_med} and for females: {rf_med}.")
        print("Observing the statistics, we conclude that we can not reject the hypothesis that the male\n"
              "population is the same as the female population for the return rates.  If one were to check\n"
              "the p-values of the the one-sided alternative hypothesises one would find that the minimum\n"
              "p-value is 0.368 which of course is also not significant.  That the medians are exactly the\n"
              "same is not surprising in light of the virtually discrete values.")
        print("\nOverall, we see therefore that all the rates calculated are likely independent of gender.\n"
              "This is not surprising given the histograms shown earlier, which even show\n"
              "that the distributions of all the product categories look similar. This is further reflected in the\n"
              "similarity of the medians which all hover around 30.4%. Of course the return rate is a different\n"
              "type of metric and we should not be surprised that its median of 42.85% is quite different than\n"
              "that of the other rates.")
        print('An interesting result that can be a candidate for future work (beyond the scope of\n '
              'gender) would to see if Wilcoxon rank-sum tests indicate that the rates are also independent of\n'
              'category.  For a synthetic dataset such as this, such a result would not be surprising because\n'
              'using the same parameters for the distributions in the source code would create this effect.')
        # add this result to the DataFrame summary:
        rate_stats.loc["Return Rate"] = [q[0], q[1], rm_med, rf_med]

        print(f"\nHere is the updated DataFrame of Stats:\n{rate_stats}")
        rate_stats.to_csv('rate_stats_complete')

    def plot_rate_hists(x: pd.DataFrame) -> pd.DataFrame:
        """Plots the histograms of the rates with the 0% and 100% data filtered out. Also returns
        a DataFrame of the data with the 0% and 100% data filtered out"""

        # make a new df without the 0 and 100% rate data
        nz_n100 = x.copy(deep=True)
        # sum of nulls verified separate source
        nz_n100[rates] = nz_n100[rates].map(lambda m: None if m == 0 or m == 100 else m)
        # plot all the histograms:
        for i in range(0, 5):
            fig = px.histogram(nz_n100, x=rates[i], color='Gender', barmode='group',
                               color_discrete_sequence=["hotpink", "blue"], template="simple_white",
                               title=f"Normalized Histogram of {titles[i]} by Individuals "
                                     f"(0 and 100% rates excluded)",
                               histnorm='percent')
            fig.show()
        print("As expected from the previous EDA, the histograms look similar. The return rate histogram which\n"
              "was not shown before looks interesting. It appears to be a superposition of the distributions for\n"
              "the different number of purchases by individuals.\n"
              "Let's look at the histograms of the number of purchases.")

        # Plot the number of purchases by all individuals
        fig1 = px.histogram(nz_n100, x='Num Purchases', color='Gender', barmode='group',
                            color_discrete_sequence=["hotpink", "blue"], template="simple_white",
                            title=f"Normalized Histogram of Number of Purchases of Individuals "
                            f"(All Data Included)",
                            histnorm='percent')
        fig1.show()

        print("We see that between 2 and 7 purchases covers a most of the individuals in the study.\n"
              "For completeness let us look at the return rates with the 0% and 100% cases included:")
        fig2 = px.histogram(df, x='Return Rate', color='Gender', barmode='group',
                            color_discrete_sequence=["hotpink", "blue"], template="simple_white",
                            title=f"Normalized Histogram of Percent of Orders That Were Returned by Individuals "
                            f"(All Data Included)",
                            histnorm='percent')
        fig2.show()

        print("\nThese data can help to explain the interesting pattern in the return rates."
              "The raw data imply that 40.59 percent of the purchases have a return.\n"
              "By making an approximation of considering contributions from binomial experiments with the number of\n"
              "trials being 2, 3, 4, 5, 6, and 7 (the number of purchases mentioned above) with p = .4059 \n"
              "and considering that the x-axis of the histogram shown earlier is the number of successes (returns)\n"
              "normalized by the number trials (purchases) we would observe a very large peak at 50%\n"
              "and peaks at 66.67% and 33.33% and the other minor peaks around 25%, 40%, 60%, and 75%.\n"
              "We note that the pattern in the return rates histogram is consistent with those observations."
              "Therefore we do not conclude that this 'interesting' pattern is unusual. \n")
        # return the DataFrame with the 0% and 100% rates nulled
        return nz_n100

    # shadowing df is OK here
    def make_zero_100(df: pd.DataFrame):
        """
        makes a DataFrame that summarizes the percentage of 100% and 0% rate data vs gender and prints it, and
        saves a csv file representation thereof
        :param df: A DataFrame of the Kaggle data under consideration
        :return: None
        """
        zero100_freqs = pd.DataFrame(index=rates, columns=['Male Zero', 'Female Zero', 'Male 100', 'Female 100'])
        male_indiv = df[df['Gender'] == 'Male']
        female_indiv = df[df['Gender'] == 'Female']
        num_males = male_indiv.shape[0]
        num_females = female_indiv.shape[0]
        print(f"\nmales: {num_males} females: {num_females}")

        for i in rates:
            # get zero percent data
            m = male_indiv[male_indiv[i] == 0].shape[0]
            f = female_indiv[female_indiv[i] == 0].shape[0]
            zero100_freqs.at[i, 'Male Zero'] = m
            zero100_freqs.at[i, 'Female Zero'] = f

            # get 100 percent data
            m = male_indiv[male_indiv[i] == 100].shape[0]
            f = female_indiv[female_indiv[i] == 100].shape[0]
            zero100_freqs.at[i, 'Male 100'] = m
            zero100_freqs.at[i, 'Female 100'] = f

        print("\nHere are the frequencies of sales of individuals by Category for the 0% and 100% cases:")
        zero100_freqs.to_csv("zero 100 freqs")
        print(zero100_freqs)
        print("\nHere are the same data but as normalized percentages")
        zero100_freqs[['Male Zero', 'Male 100']] = zero100_freqs[['Male Zero', 'Male 100']]/num_males * 100
        zero100_freqs[['Female Zero', 'Female 100']] = zero100_freqs[['Female Zero', 'Female 100']] / num_females * 100
        zero100_freqs.to_csv("zero 100 rate pct")
        print(zero100_freqs)
        print("\nWe see from a practical standpoint that these extreme case 0 and 100 rates\n"
              "are quite close to each other with respect to gender. Let us now proceed to look at the "
              "distributions of these rates with the 0 and 100 percent rates removed.")

    # EDA of the rate data
    print("\n We now look at the numeric rate data:")
    rate_columns = ['Gender', 'Book pct', 'Clothes pct', 'Home pct', 'Electron pct']
    all_r = df[rate_columns]
    all_r.hist(by='Gender', bins=40)
    show()
    print("\nThe histograms just shown and percentiles below show what it is convenient to consider\n"
          "3 cases: 0% and 100% which have significant probabilities probably especially for small orders \n"
          "and the rest of the values.")
    print(all_r.describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.95, 0.97, 0.99]))

    # create dataframe of 100% and 0% rate data by gender and show the results (does not mutate the DataFrame df)
    make_zero_100(df)
    # make the 0 and 100% data in df null and copy to a new df and use for histogram plots in a for loop
    # (it does not mutate the DataFrame passed to it, but creates a deep copy and modifies the copy and returns
    # that DataFrame
    # Plot cum dist without the 0% and 100% special case extrema:

    zero100 = plot_rate_hists(df)
    zero100.to_csv("zero100DataFrame")

    # perform non-parametric test on the data without the 100% and 0% points
    # by doing Wilcoxon rank-sum tests.
    # XXX check whether this will mutate zero100
    wilcoxon_rank_sum(zero100)
