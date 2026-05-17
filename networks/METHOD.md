## Occupational Networks

The following occupational networks build on work presented by Mealy (2018), who created a labour flow network and skill similarity network of occupations in the United States. In their paper, they show that the overlap between occupational tasks is positively correlated with the probability of transitioning between occupations. They found that, of all occupational attributes in the O*NET database, intermediate work activities could explain the most variance in occupational transitions. For the United States, overlap in intermediate work activities could account for 9% of the variance in occupational transitions.

I build on this work by creating a labour flow network for the UK and mapping O*NET's intermediate work activities to the UK's SOC occupational taxonomy. I also create occupational networks that capture geographic and industrial similarity, as well as wage and unemployment differences. These additional factors can help explain additional variance in occupational transitions.

The two primary datasets I use to construct the occupation networks in this thesis are the UK Household Longitudinal Study and the O*NET database. In addition to these sources, I use the Labour Market Information for All API, developed by the UK Department for Education, to get yearly estimates for occupational unemployment derived from the UK Labour Force Survey.

The UKHLS is a comprehensive study of approximately 40,000 households that has been administered annually throughout the UK since 2009. In particular, I use Waves 3–12, covering 2011–2022, of unweighted data for individuals over the age of 16 to construct networks for occupational labour flows ($L$), geographic similarity ($G$), industrial similarity ($N$), and wage differences ($W$).

O*NET is a database developed by the US Department of Labor with a comprehensive list of occupations, related working activities, and other details. It has been used extensively to better understand the relationship between occupations, skills, and shifts in labour demand. I link O*NET occupations to the UK's Standard Occupational Classification (SOC) codes to create a skill similarity network ($S$), using the mapping provided by the Labour Market for All API.

All of these networks, as well as the pairwise differences in unemployment ($U$) and wage ($W$), are built with SOC 3-digit occupations as their nodes. The labour flow network is used to fit the labour flow model from del Rio-Chanon (2021), while the skill, geography, industry, and wage-difference networks are used to represent transition costs in the heuristic function for determining the optimal retraining policy. Additionally, I regress the latter networks and the unemployment data on the labour flow network to determine which factors are most strongly correlated with occupational transitions.

### Labour Flow Network

The labour flow network ($L$) was created following the approach presented by del Rio-Chanon (2021), which is an extension of the method from Mealy (2018). Here, I use the UKHLS dataset to get the transition probabilities between each pair of occupations:

$$
P_{ij} = \frac{T_{ij}}{\sum_j T_{ij}}
$$

where $T_{ij}$ is the total number of switches from occupation $i$ to occupation $j$ for all pairs of occupations where $i \neq j$. $P_{ij}$ can be interpreted as the yearly probability of switching from occupation $i$ to occupation $j$, given that an individual is not staying in the same occupation.

To estimate the probability of which occupation an individual applies to when they change jobs, we take:

$$
L_{ij} =
\begin{cases}
r & \text{if } i = j \\
P_{ij}(1-r) & \text{if } i \neq j
\end{cases}
$$

where $r$ is the estimated probability that an individual applies within their occupation when switching jobs in a single time step. I estimate $r$ following the approach in del Rio-Chanon (2021), which assumes $r$ is the same across time and occupations.

The probability that a worker stays in their occupation in a given year is:

$$
p = ((1-u) + ur)^y
$$

where $u$ is the probability of being unemployed and $y$ is the number of time steps in a year. Solving for $r$ gives:

$$
r = \frac{p^{1/y} + u - 1}{u}
$$

We calculate $p$ from the UKHLS dataset by taking the proportion of individuals who stayed in their occupation each year. I find $p = 0.8052$, average unemployment between 2011 and 2021 was $u = 0.0571$, and $y = 52 / 7.4286$ since there are 7.4286 weeks per time step; see Section \ref{calibration}. Hence, $r = 0.4665$.

### Skill Similarity Network

I created the skill network following the approach in Mealy (2018), but first linked the intermediate working activities of the O*NET occupations to the best SOC 4-digit occupation match using the Labour Market Info for All API's O*NET-to-SOC matching endpoint. If at least one 4-digit occupation had a given intermediate working activity, that activity was included in the 3-digit aggregation.

$$
S_{ij} =
\min\left(
\frac{\sum_w B_{iw} B_{jw} r_w}{\sum_w B_{iw}},
\frac{\sum_w B_{iw} B_{jw} r_w}{\sum_w B_{jw}}
\right)
$$

where $B$ is a bipartite network of 3-digit occupations and skills. $B_{iw} = 1$ if the SOC 3-digit occupation $i$ has at least one 4-digit occupation with work activity $w$; otherwise, $B_{iw} = 0$. The term $r_w$ weights how rare each work activity is across 3-digit occupations:

$$
r_w = \frac{1}{\sum_i B_{iw}}
$$

The measure presented here is identical to the one initially presented by Zaccaria (2014), who used it to relate products by the countries that exported them. The rarity adjustment, $r_w$, gives more weight to rare skills, since sharing a rare skill is a better indicator of occupational similarity than sharing a common skill. Mealy (2018) found that this rarity-adjusted measure better explains transition probabilities.

### Geographic Similarity Network

The geography network ($G$) is also constructed from the UKHLS data. Occupation groups are associated with one of 12 Government Office Regions by the number of individuals in occupation $i$ who work in region $r$ across all years. This means individuals are counted multiple times, even if they stay in the same occupation. I denote this count as $v_{ir}$.

First, a bipartite network relating occupations and regions is created:

$$
B_{ir} = \frac{v_{ir}}{\sum_i v_{ir}}
$$

In $B_{ir}$, each value can be interpreted as the percentage of jobs in region $r$ that are in occupation $i$. To create the geographic similarity network, I take the cosine similarity between each occupation row in $B_{ir}$:

$$
G_{ij} = \frac{B_i \cdot B_j}{\|B_i\| \|B_j\|}
$$

Since this network is based on only 12 regions, there is high similarity between occupations, with most values of $G_{ij} > 0.8$. However, within regions, there is likely to be significant heterogeneity in the distribution of occupations, which would limit occupational transitions. For instance, Bristol and Cornwall are both in the South West region, but they likely have very different occupations, and it would be costly to move between them. Therefore, future work should use a more granular geographic distribution of occupations.

### Industrial Similarity Network

The industry network ($N$) is constructed similarly to the geographic network. Occupation groups are associated with 88 SIC industry divisions by the number of individuals in occupation $i$ who work in industry $n$ across all years. This means individuals are counted multiple times, even if they stay in the same occupation. I denote this count as $v_{in}$.

First, a bipartite network relating occupations and industries is created:

$$
B_{in} = \frac{v_{in}}{\sum_i v_{in}}
$$

In $B_{in}$, each value can be interpreted as the percentage of jobs in industry $n$ that are in occupation $i$. To create the industrial similarity network, I take the cosine similarity between each occupation row in $B_{in}$:

$$
N_{ij} = \frac{B_i \cdot B_j}{\|B_i\| \|B_j\|}
$$

Since this network is based on 88 SIC sectors, there is substantial heterogeneity across occupations, and most occupations are dissimilar, with most values of $N_{ij} < 0.2$.

### Pairwise Wage Differences

Wages are derived from the UKHLS dataset by taking the average wage observed for each individual $p \in P_{it}$ in occupation $i$ at time $t$, across all years. This means individuals are counted multiple times, even if they stay in the same occupation. We denote the wage of an individual $p$ at time $t$ as $w_{pt}$.

$$
W_{ij} =
\frac{1}{T}
\sum_{t=0}^{T}
\left(
\sum_{p \in P_{jt}} \frac{w_{pt}}{\|P_{jt}\|}-
\sum_{p \in P_{it}} \frac{w_{pt}}{\|P_{it}\|}
\right)
$$

where $P_{it}$ is the set of individuals at time $t$ who are in occupation $i$. This network is then min-max normalized to fit on $[0,1]$ like the rest of the networks.

Since the network is symmetric, meaning $W_{ij} = -W_{ji}$, $W_{ij} = 0.5$ implies that the origin occupation $i$ and target occupation $j$ have the same wage levels. When $W_{ij} < 0.5$, the origin occupation $i$ has higher wages; when $W_{ij} > 0.5$, the target occupation $j$ has higher wages.

### Pairwise Unemployment Differences

For pairwise differences in unemployment, I get the yearly unemployment rates for each SOC 3-digit occupation from the Labour Market for All API, which are estimated from the Labour Force Survey. I denote this rate as $u_{it}$.

$$
U_{ij} =
\frac{1}{T}
\sum_{t=0}^{T}
(u_{jt} - u_{it})
$$

where $T$ is the number of years for which we have data for both $u_i$ and $u_j$. The final matrix is then normalized by dividing by the maximum absolute value in $U_{ij}$.

Similar to the wage differences, this is min-max normalized to fit on $[0,1]$ and has a similar interpretation to the wage network. When $U_{ij} < 0.5$, the origin occupation $i$ has higher unemployment; when $U_{ij} > 0.5$, the target occupation $j$ has higher unemployment.
