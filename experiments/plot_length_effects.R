library(tidyverse)

setwd("/home/qy2672/learning-alternations")
model_data <- read.csv("experiments/long_first_regression.csv")
model_data_long <- model_data %>%
  filter(long_first == 0.5)
model_data_short <- model_data %>%
  filter(long_first == -0.5)

model_data_long %>%
  ggplot(aes(x=score, fill=interaction(recipient_pronoun > 0, length_difference >= 0))) +
  geom_histogram(alpha=0.5, position="identity") +
  ggtitle("Long-First Condition") +
  scale_fill_discrete(name="Conditions",
                      labels=c("Non-Pronoun & Short", "Non-Pronoun & Long",
                              "Pronoun & Short", "Pronoun & Long"))

model_data_short %>%
  ggplot(aes(x=score, fill=interaction(recipient_pronoun > 0, length_difference >= 0))) +
  geom_histogram(alpha=0.5, position="identity") +
  ggtitle("Short-First Condition") +
  scale_fill_discrete(name="Conditions",
                      labels=c("Non-Pronoun & Short", "Non-Pronoun & Long",
                              "Pronoun & Short", "Pronoun & Long"))

