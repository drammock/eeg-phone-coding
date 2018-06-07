---
title: >
  Supplement to “Learning phonological features from EEG recordings during speech perception: three feature systems compared”
author:
- name: Daniel R. McCloy
- name: Adrian K. C. Lee
  email: akclee@uw.edu
  affiliation:
  - Institute for Learning and Brain Sciences, University of Washington
documentclass: article
classoption: oneside
fontsize: 12pt
geometry:
- letterpaper
- margin=2cm
biblio-style: jasasty-ay-web
bibliography: bib/supplement.bib
---

# Matrix diagonality

Here we describe a method of computing the “diagonality” of a matrix.  The
measure is based on the Pearson correlation for points in the $(x,y)$ plane,
and when applied to a non-negative square matrix, yields values between −1 and
1. To give an intuition of what the results of this diagonality measure will
look like: diagonal matrices yield a value of 1, matrices that are nonzero only
on the *minor diagonal* yield a value of −1, and uniform matrices yield a value
of 0.  The line of reasoning follows and expands on a discussion posted on
Mathematics Stack Exchange [@mathexch] for computing the correlation between
the rows and columns of a square matrix.

Consider a non-negative square matrix $A$:

$$A = \begin{bmatrix}
0   & 0   & 0   & 0.4 \\
0   & 0   & 0.2 & 0   \\
0   & 0   & 0   & 0   \\
0.3 & 0   & 0   & 0   \\
\end{bmatrix}$$

We can conceive of $A$ as representing a set of points $(c,r)$, where the
coordinates of those points are determined by the column and row indices of the
matrix (so for a 4×4 matrix like $A$, the $(c,r)$ locations are given by
$c,r \in \{1,2,3,4\}$). For each of those points $(c,r)$, the corresponding
entry in $A$ describes the mass of the point at that location.  Since 13 of the
16 entries in $A$ are zero, we can represent $A$ in the $c,r$ plane as a set of
just 3 points of differing masses.  This is shown in figure \ref{anticorr},
with grayscale values and annotations denoting the mass at each location:

\begin{SCfigure}[50][h]
\includegraphics{sfigs/grid-a}
\caption{Representation of matrix $A$ in the $c,r$ plane. Note that $r$ increases downwards, to accord with conventional matrix row numbering.}
\label{anticorr}
\end{SCfigure}

Ordinarily, Pearson correlations are computed on a set of points that are
equally weighted, so to compute the Pearson correlation for the points in
figure \ref{anticorr} we must either incorporate weights into the Pearson
formula, or transform our data such that each point is equally weighted.  The
latter approach can be achieved straighforwardly by scaling our matrix $A$ such
that all its entries are integers, and treating those integer entries as the
number of replicated, equally-weighted points at each location.  This scaling
factor $s$ will be the least common denominator of the fractional
representations of the elements of $A$, which in this case is 10.[^lcd] Thus we can compute $B = sA$:

$$B = s \times A = \begin{bmatrix}
0 & 0 & 0 & 4 \\
0 & 0 & 2 & 0 \\
0 & 0 & 0 & 0 \\
3 & 0 & 0 & 0 \\
\end{bmatrix}$$

[^lcd]: Here the entries of $A$ have been chosen to make the scale factor small
(so that the set of points $P$ is of manageable size for illustrative purposes)
but in general there is no need for the entries of $A$ to be “tidy” numbers,
because, as will become clear below, the scaling factor $s$ will cancel out of
the Pearson equation, and we can operate directly on the matrix $A$ without
affecting the resulting correlation value.

From $B$ then we can generate a set of (equally weighted) points $P$, where the coordinates of each point are $(column, row)$ indices, and the number of replicates of each coordinate pair is determined by $B_{r,c}$: $P = \begin{bmatrix}(1,4), (1,4), (1,4), (3,2), (3,2), (4,1), (4,1), (4,1), (4,1)\end{bmatrix}$

A plot of the points $P$ will superficially resemble figure \ref{anticorr},
because the replicated points will be overplotted and impossible to
distinguish.  Nonetheless, we can now straightforwardly apply the formula for
the Pearson correlation of a sample, given here with coordinates $(c,r)$
replacing the more traditional $(x,y)$:

\begin{align}
\hat{\rho} =& \frac{\mathrm{cov}(c,r)}{\mathrm{stdev}(c) \times \mathrm{stdev}(r)} \\
           =& \frac{\sum_{i=1}^n (c_i - \bar{c})(r_i - \bar{r})}{\sqrt{\sum_{i=1}^n (c_i - \bar{c})^2} \times \sqrt{\sum_{i=1}^n (r_i - \bar{r})^2}} \label{pearson2} \\
           =& \frac{n \sum_{i=1}^n c_i r_i - \sum_{i=1}^n c_i \sum_{i=1}^n r_i}{\sqrt{n \sum_{i=1}^n c_i^2 - \left( \sum_{i=1}^n c_i \right)^2}\sqrt{n \sum_{i=1}^n r_i^2 - \left( \sum_{i=1}^n r_i \right)^2}}\label{pearson3}
\end{align}

The derivation of equation (\ref{pearson3}) from equation (\ref{pearson2}) is
provided in a following section, for those interested in the details. We
can now translate equation (\ref{pearson3}) into a form that more directly
reflects our matrix $B$: the term $n$ is the number of points in $P$, which is
also equal to the sum of the entries of $B$, and the coordinates $c_i$ and
$r_i$ are the column and row indices of each (nonzero) entry of the matrix.
For an *m*×*m* matrix, if we define a vector
$i = \begin{bmatrix} 1 \dots m \end{bmatrix}$ representing row or column
indices, and a vector $w$ of all ones of the same length as $i$, then we can
rewrite the summation terms in equation (\ref{pearson3}) as follows:

\begin{alignat*}{5}
n            &= wBw^T & \qquad & \qquad & \sum r_i &= iBw^T & \qquad & \qquad & \sum r_i^2 &= i^2Bw^T \\
\sum r_i c_i &= iBi^T & \qquad & \qquad & \sum c_i &= wBi^T & \qquad & \qquad & \sum c_i^2 &= wB(i^2)^T
\end{alignat*}

This yields the following equivalent expression for $\hat{\rho}$:

\begin{equation}
\hat{\rho} = \frac{wBw^T \times iBi^T - iBw^T \times wBi^T}{\sqrt{wBw^T \times i^2Bw^T - (iBw^T)^2} \times \sqrt{wBw^T \times wB(i^2)^T - (wBi^T)^2}} \label{linalg}
\end{equation}

Note here that if we substitute $sA$ for $B$ in equation (\ref{linalg}),
the scaling factor $s$ can be factored out of both numerator and
denominator and cancelled, showing that it is not in fact
necessary to scale the matrix to integer values in order to compute the
correlation between rows and columns using the Pearson formula.
Rather, the original values of $A$ act as (fractional)
weights on the vectors of row and column indices, in exactly the same
fashion as the integer numbers of replicated, superimposed points in the
scaled matrix $B$ did:

\begin{align}
\hat{\rho} =& \frac{ w(sA)w^T \times i(sA)i^T -  i(sA)w^T \times w(sA)i^T}{           \sqrt{w(sA)w^T  \times i^2(sA)w^T - (i(sA)w^T)^2} \times \sqrt{w(sA)w^T  \times w(sA)(i^2)^T - (w(sA)i^T)^2}} \\
           =& \frac{s^2(wAw^T \times iAi^T)   - s^2(iAw^T \times   wAi^T)}{           \sqrt{s^2(wAw^T \times i^2Aw^T    - (iAw^T)^2)}   \times \sqrt{s^2(wAw^T \times wA(i^2)^T    - (wAi^T)^2)}} \\
           =& \frac{s^2(wAw^T \times iAi^T    -     iAw^T \times   wAi^T)}{s^2 \left( \sqrt{wAw^T     \times i^2Aw^T    - (iAw^T)^2}    \times \sqrt{wAw^T     \times wA(i^2)^T    - (wAi^T)^2} \right)} \\
           =& \frac{    wAw^T \times iAi^T    -     iAw^T \times   wAi^T }{           \sqrt{wAw^T     \times i^2Aw^T    - (iAw^T)^2}    \times \sqrt{wAw^T     \times wA(i^2)^T    - (wAi^T)^2}} \label{linalg2}
\end{align}

Each of the terms expressed as matrix multiplications in equation
(\ref{linalg2}) can also be equivalently expressed as a combination of sums,
outer products ($\otimes$) and Hadamard products ($\circ$); the motivation for
this will become clear below.  Summarizing the equivalences:

\begin{alignat*}{3}
n \enskip &= \enskip wAw^T \enskip &=& \enskip \sum_{r,c} A_{r,c} \\
\sum r_i \enskip &= \enskip iAw^T \enskip &=& \enskip
    \sum_{r,c} (i \otimes w \circ A)_{r,c} \\
\sum c_i \enskip &= \enskip wAi^T \enskip &=& \enskip
    \sum_{r,c} (w \otimes i \circ A)_{r,c} \\
\sum r_i c_i \enskip &= \enskip iAi^T \enskip &=& \enskip
    \sum_{r,c} (i \otimes i \circ A)_{r,c} \\
\sum r_i^2 \enskip &= \enskip i^2Aw^T \enskip &=& \enskip
    \sum_{r,c} (i^2 \otimes w \circ A)_{r,c} \\
\sum c_i^2 \enskip &= \enskip wA(i^2)^T \enskip &=& \enskip
    \sum_{r,c} (w \otimes i^2 \circ A)_{r,c}
\end{alignat*}

We can now express $\hat{\rho}$ as:

\begin{equation}\textstyle
\hat{\rho} = \frac{\sum\limits_{r,c} A_{r,c} \; \times \; \sum\limits_{r,c} (i \,\otimes i \,\circ A)_{r,c} \enskip - \enskip \sum\limits_{r,c} (i \,\otimes w \,\circ A)_{r,c} \; \times \; \sum\limits_{r,c} (w \,\otimes i \,\circ A)_{r,c}}{\sqrt{\sum\limits_{r,c} A_{r,c} \; \times \; \sum\limits_{r,c} (i^2 \,\otimes w \,\circ A)_{r,c} \; - \; \left(\sum\limits_{r,c} (i \,\otimes w \,\circ A)_{r,c} \right)^2} \; \times \; \sqrt{\sum\limits_{r,c} A_{r,c} \; \times \; \sum\limits_{r,c} (w \,\otimes i^2 \,\circ A)_{r,c} \; - \; \left(\sum\limits_{r,c} (w \,\otimes i \,\circ A)_{r,c} \right)^2}}
\end{equation}

Visually, these computations can be represented as Hadamard products of the
matrix $A$ with various weighting matrices reflecting combinations of the row
and column indices (although it is more accurate to think of the matrix $A$
itself serving as the weights, and the row and column indices are the “real”
data points whose correlation is being computed):

$$\frac{\sum A \times \sum \left( \overset{i \,\otimes\,
i}{\img{sfigs/mats/weight-rowcol}} \circ A \right) - \sum \left( \overset{i
\,\otimes\, w}{\img{sfigs/mats/weight-row}} \circ A \right) \times \sum \left(
\overset{w \,\otimes\, i}{\img{sfigs/mats/weight-col}} \circ A
\right)}{\sqrt{\sum A \times \sum \left( \overset{i^2 \,\otimes\,
w}{\img{sfigs/mats/weight-rowsqu}} \circ A \right) - \left(\sum \left(
\overset{i \,\otimes\, w}{\img{sfigs/mats/weight-row}} \circ A \right)
\right)^2} \times \sqrt{\sum A \times \sum \left( \overset{w \,\otimes\,
i^2}{\img{sfigs/mats/weight-colsqu}} \circ A \right) - \left(\sum \left(
\overset{w \,\otimes\, i}{\img{sfigs/mats/weight-col}} \circ A \right)
\right)^2}}$$

Example computations follow with different kinds of matrices substituted for
$A$:

## Diagonal matrix:

$$\textstyle \frac{\sum \left( \img{sfigs/mats/diagonal} \right) \times \sum
\left( \img{sfigs/mats/weight-rowcol} \,\circ\, \img{sfigs/mats/diagonal}
\right) - \sum \left( \img{sfigs/mats/weight-row} \,\circ\,
\img{sfigs/mats/diagonal} \right) \times \sum \left(
\img{sfigs/mats/weight-col} \,\circ\, \img{sfigs/mats/diagonal}
\right)}{\sqrt{\sum \left( \img{sfigs/mats/diagonal} \right) \times \sum \left(
\img{sfigs/mats/weight-rowsqu} \,\circ\, \img{sfigs/mats/diagonal} \right) -
\left(\sum \left( \img{sfigs/mats/weight-row} \,\circ\,
\img{sfigs/mats/diagonal} \right) \right)^2} \times \sqrt{\sum \left(
\img{sfigs/mats/diagonal} \right) \times \sum \left(
\img{sfigs/mats/weight-colsqu} \,\circ\, \img{sfigs/mats/diagonal} \right) -
\left(\sum \left( \img{sfigs/mats/weight-col} \,\circ\,
\img{sfigs/mats/diagonal} \right) \right)^2}} = 1.0$$

## Tridiagonal matrix:

$$\textstyle \frac{\sum \left( \img{sfigs/mats/tridiagonal} \right) \times \sum
\left( \img{sfigs/mats/weight-rowcol} \,\circ\, \img{sfigs/mats/tridiagonal}
\right) - \sum \left( \img{sfigs/mats/weight-row} \,\circ\,
\img{sfigs/mats/tridiagonal} \right) \times \sum \left(
\img{sfigs/mats/weight-col} \,\circ\, \img{sfigs/mats/tridiagonal}
\right)}{\sqrt{\sum \left( \img{sfigs/mats/tridiagonal} \right) \times \sum
\left( \img{sfigs/mats/weight-rowsqu} \,\circ\, \img{sfigs/mats/tridiagonal}
\right) - \left(\sum \left( \img{sfigs/mats/weight-row} \,\circ\,
\img{sfigs/mats/tridiagonal} \right) \right)^2} \times \sqrt{\sum \left(
\img{sfigs/mats/tridiagonal} \right) \times \sum \left(
\img{sfigs/mats/weight-colsqu} \,\circ\, \img{sfigs/mats/tridiagonal} \right) -
\left(\sum \left( \img{sfigs/mats/weight-col} \,\circ\,
\img{sfigs/mats/tridiagonal} \right) \right)^2}} = 0.806$$

## Uniform matrix:

$$\textstyle \frac{\sum \left( \img{sfigs/mats/uniform} \right) \times \sum
\left( \img{sfigs/mats/weight-rowcol} \,\circ\, \img{sfigs/mats/uniform}
\right) - \sum \left( \img{sfigs/mats/weight-row} \,\circ\,
\img{sfigs/mats/uniform} \right) \times \sum \left( \img{sfigs/mats/weight-col}
\,\circ\, \img{sfigs/mats/uniform} \right)}{\sqrt{\sum \left(
\img{sfigs/mats/uniform} \right) \times \sum \left(
\img{sfigs/mats/weight-rowsqu} \,\circ\, \img{sfigs/mats/uniform} \right) -
\left(\sum \left( \img{sfigs/mats/weight-row} \,\circ\,
\img{sfigs/mats/uniform} \right) \right)^2} \times \sqrt{\sum \left(
\img{sfigs/mats/uniform} \right) \times \sum \left(
\img{sfigs/mats/weight-colsqu} \,\circ\, \img{sfigs/mats/uniform} \right) -
\left(\sum \left( \img{sfigs/mats/weight-col} \,\circ\,
\img{sfigs/mats/uniform} \right) \right)^2 }} = 0.0$$

## Random matrix:

$$\textstyle \frac{\sum \left( \img{sfigs/mats/random} \right) \times \sum
\left( \img{sfigs/mats/weight-rowcol} \,\circ\, \img{sfigs/mats/random}
\right) - \sum \left( \img{sfigs/mats/weight-row} \,\circ\,
\img{sfigs/mats/random} \right) \times \sum \left( \img{sfigs/mats/weight-col}
\,\circ\, \img{sfigs/mats/random} \right)}{\sqrt{\sum \left(
\img{sfigs/mats/random} \right) \times \sum \left(
\img{sfigs/mats/weight-rowsqu} \,\circ\, \img{sfigs/mats/random} \right) -
\left(\sum \left( \img{sfigs/mats/weight-row} \,\circ\, \img{sfigs/mats/random}
\right) \right)^2} \times \sqrt{\sum \left( \img{sfigs/mats/random} \right)
\times \sum \left( \img{sfigs/mats/weight-colsqu} \,\circ\,
\img{sfigs/mats/random} \right) - \left(\sum \left( \img{sfigs/mats/weight-col}
\,\circ\, \img{sfigs/mats/random} \right) \right)^2}} = 0.119$$

## Minor diagonal matrix:

$$\textstyle \frac{\sum \left( \img{sfigs/mats/antidiagonal} \right) \times
\sum \left( \img{sfigs/mats/weight-rowcol} \,\circ\,
\img{sfigs/mats/antidiagonal} \right) - \sum \left( \img{sfigs/mats/weight-row}
\,\circ\, \img{sfigs/mats/antidiagonal} \right) \times \sum \left(
\img{sfigs/mats/weight-col} \,\circ\, \img{sfigs/mats/antidiagonal}
\right)}{\sqrt{\sum \left( \img{sfigs/mats/antidiagonal} \right) \times \sum
\left( \img{sfigs/mats/weight-rowsqu} \,\circ\, \img{sfigs/mats/antidiagonal}
\right) - \left(\sum \left( \img{sfigs/mats/weight-row} \,\circ\,
\img{sfigs/mats/antidiagonal} \right) \right)^2} \times \sqrt{\sum \left(
\img{sfigs/mats/antidiagonal} \right) \times \sum \left(
\img{sfigs/mats/weight-colsqu} \,\circ\, \img{sfigs/mats/antidiagonal}
\right) - \left(\sum \left( \img{sfigs/mats/weight-col} \,\circ\,
\img{sfigs/mats/antidiagonal} \right) \right)^2}} = -1$$


# Derivation of the second form of the Pearson equation

Repeating equation (\ref{pearson2}) here, but with more familiar $(x,y)$
coordinates in place of the $(c,r)$ notation used in equation (\ref{pearson2}):

\begin{equation}
\hat{\rho} = \frac{\sum\limits_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum\limits_{i=1}^n (x_i - \bar{x})^2} \times \sqrt{\sum\limits_{i=1}^n (y_i - \bar{y})^2}} \label{pearson2repeat}
\end{equation}

We begin by examining only the numerator:

\begin{alignat}{2}
& \text{original numerator} & \quad & \sum\limits_{i=1}^n (x_i - \bar{x})(y_i - \bar{y}) \\
& \text{after multiplying the binomial terms} & \quad & \sum\limits_{i=1}^n (x_i y_i - x_i \bar{y} - \bar{x} y_i + \bar{x} \bar{y}) \\
& \text{after distributing the summation} & \quad & \sum x_i y_i - \sum x_i \bar{y} - \bar{x} \sum y_i + \bar{x} \bar{y} n \\
& \text{after rewriting $\bar{x}$ and $\bar{y}$} & \quad & \sum x_i y_i - \sum x_i \frac{\sum y_i}{n} - \frac{\sum x_i}{n} \sum y_i + \frac{\sum x_i}{n} \frac{\sum y_i}{n} n \\
& \text{after canceling $n$ in final term} & \quad & \sum x_i y_i - \sum x_i \frac{\sum y_i}{n} - \frac{\sum x_i}{n} \sum y_i + \frac{\sum x_i}{n} \sum y_i \\
& \text{after canceling last two terms} & \quad & \sum x_i y_i - \sum x_i \frac{\sum y_i}{n} \\
& \text{after multiplying by $\frac{n}{n}$} & \quad & \frac{1}{n} \left( n \sum x_i y_i - \sum x_i \sum y_i \right)
\end{alignat}

Since the contents of each squareroot in the denominator of equation
(\ref{pearson2repeat}) have the same form as the original numerator, the same
sequence of steps will convert the standard deviation along the x dimension
$\sqrt{\sum (x_i - \bar{x})^2}$ into
$\sqrt{\frac{1}{n} \left( \sum x_i^2 - \left( \sum x_i \right)^2 \right)}$,
and likewise for the corresponding $y$ term, yielding:

\begin{equation}
\hat{\rho} = \frac{\frac{1}{n} \left( n \sum x_i y_i - \sum x_i \sum y_i \right)}{\sqrt{\frac{1}{n} \left( n \sum x_i^2 - \left( \sum x_i \right)^2 \right)}\sqrt{\frac{1}{n} \left( n \sum y_i^2 - \left( \sum y_i \right)^2 \right)}}
\end{equation}

and after cancelling the $\frac{1}{n}$ terms, we have reached the form of equation (\ref{pearson3}):

\begin{equation}
\hat{\rho} = \frac{n \sum x_i y_i - \sum x_i \sum y_i}{\sqrt{n \sum x_i^2 - \left( \sum x_i \right)^2}\sqrt{n \sum y_i^2 - \left( \sum y_i \right)^2}}
\end{equation}

# Supplementary figures

## Signal-to-noise ratio

One potential pitfall when analyzing neurophysiological data is variability in signal-to-noise ratio (SNR) across subjects (this can be especially problematic for group comparisons between patient and control populations).  To ensure that SNR was not the driving factor in our measure of matrix diagonality, we performed some basic analyses of SNR in our data.

\begin{SCfigure}[5][h]
\includegraphics[width=0.75\textwidth]{sfigs/subject-summary}
\caption{Plots of SNR, blink detection, and retained epochs for each subject.  The total number of trials varies slightly for each subject due to additional (unanalyzed) trials from foreign talkers; each listener heard only 2 of the 4 foreign talkers, and the languages represented by the foreign talkers did not necessarily have equal numbers of consonants.}
\end{SCfigure}

\phantom{x}

\begin{figure}
\includegraphics{sfigs/snr-vs-matrix-diagonality}
\caption{SNR versus matrix diagonality for each phonological feature system. There is no evidence to support a correlation between SNR and matrix diagonality in any of the three feature systems (all p-values > 0.05, uncorrected).}
\end{figure}

## Confusion matrices for individual subjects

\begin{figure}
\includegraphics{sfigs/fig-psa}
\caption{Confusion matrices for each subject for the PSA phonological feature system classifiers.}
\end{figure}

\begin{figure}
\includegraphics{sfigs/fig-spe}
\caption{Confusion matrices for each subject for the SPE phonological feature system classifiers.}
\end{figure}

\begin{figure}
\includegraphics{sfigs/fig-phoible}
\caption{Confusion matrices for each subject for the PHOIBLE phonological feature system classifiers.}
\end{figure}

# References

\setlength{\parindent}{-0.25in}
\setlength{\leftskip}{0.25in}
\noindent
