# Executive Summary

This segmentation uses KMeans on Age, Annual Income, and Spending Score to identify 6 consumer segments with distinct demographic and behavioral profiles. The solution was selected based on silhouette diagnostics while balancing interpretability and business actionability.

## Key Findings

- 6 segments were selected, offering a balance between meaningful differentiation and stable cluster structure. The KMeans silhouette score is highest at k=6, and bootstrap stability testing reports adjusted Rand indices near 0.99 across repeated runs.
- Segments vary by income, spending intent, age, and gender. This enables both value-based targeting and customer lifecycle messaging.
- The largest segment is **Mature Value Seekers**, followed by **Premium Momentum Spenders** and balanced/established middle-income groups.

## Recommended Segment Priorities

- Prioritize **Premium Momentum Spenders** for high-value retention and premium offers. Their high spend and high income indicate strong revenue opportunity.
- Invest in **Affluent Low-Engagement Professionals** with personalized convenience and cross-sell campaigns, as they have strong financial capacity but currently low engagement.
- Cultivate **Emerging Mid-Income Explorers** through discovery campaigns and first-time experience offers, as they are a promising growth segment.
- Maintain efficient retention programs for **Mature Value Seekers** and **Established Comfort Shoppers**; these segments are stable but price- and value-sensitive.

## Strategic Recommendations

- Use **income and spending score** signals to personalize promotion intensity: premium goods for high-income/high-spend customers, discounts and loyalty drivers for lower-income/value-oriented customers.
- Align messaging with segment motivations: exclusivity and convenience for high-income segments, affordability and trust for lower-income segments, and engagement opportunities for younger explorers.
- Deploy digital channels for younger, high-interest segments and mix email plus direct outreach for mature value-oriented shoppers.

## Execution Notes

- Because this dataset is modest in size and feature scope, further validation with transactional and categorical purchase data is recommended before scaling the segmentation program.