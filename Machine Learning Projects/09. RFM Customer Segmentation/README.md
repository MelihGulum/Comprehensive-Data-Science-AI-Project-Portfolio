# RFM Customer Segmentation

## Overview

This project focuses on segmenting customers based on their purchasing behavior using RFM (Recency, Frequency, Monetary) analysis. The idea is simple: understand how recently, how often, and how much customers spend — then use that to group them in a meaningful way.

---

## Why this matters

Not all customers are the same. Some buy frequently, some haven’t returned in a while, and some contribute most of the revenue. Being able to separate these groups helps businesses:

* target the right audience
* improve retention
* design better marketing campaigns

---

## Dataset

The dataset includes customer-level purchase information along with campaign-related features.

Main variables used:

* Recency → days since last purchase
* Frequency → number of purchases
* Monetary → total spending

---

## What I did

### 1. Data preparation

* Cleaned the data and handled missing values
* Combined product-level spending into a single total spending feature

### 2. RFM calculation

* Built Recency, Frequency, and Monetary metrics for each customer

### 3. Scoring

* Used quantiles to assign scores (1–5)
* Lower recency = better score
* Higher frequency and monetary = better score

### 4. Segmentation

* Combined RFM scores into groups like:

  * Champions
  * Loyal customers
  * At risk
  * Hibernating

---

## Results & observations

* A small group of customers contributes a large portion of the revenue
* There are clear “at risk” users who haven’t engaged recently
* Some users show potential but are not fully loyal yet

---

## How this could be used

* Retarget inactive users with campaigns
* Reward high-value customers
* Focus marketing budget on segments with the highest return

---

## Notes

This is a rule-based segmentation approach. It can be extended with clustering methods like K-Means for a more data-driven grouping.

