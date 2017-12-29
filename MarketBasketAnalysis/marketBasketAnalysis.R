# clear the stored variable & plot area
rm(list = ls())
dev.off()

# load necessary libraries
library(arules)
library(arulesViz)

# Set working directory
setwd("~/Sem_4/DS/Assignment/A1/output")

# Read data
data <- read.transactions("basket.csv", format = "basket", sep=',')
inspect(data)

# Plot the item frequency plot
itemFrequencyPlot(data, topN = 20, horiz = TRUE)
itemFrequencyPlot(data, support = 0.06, type = "relative", horiz = TRUE)

# finding rules
rules <- apriori(data, parameter = list(support=0.02, confidence = 0.8))
inspect(rules)
# top 5 rules sorted in descending order of lift
sorted_rules <- sort(rules, by = "lift", decreasing = TRUE)
inspect(head(sorted_rules,5))

# plotting top 8 rules sorted in descending order of lift
top_eight <- head(sort(rules, by = "lift", decreasing = TRUE),8)
inspect(top_eight)
# plot top 8 rules
plot(top_eight, method = "graph",interactive = TRUE, shading = NA)
