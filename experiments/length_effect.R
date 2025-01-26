
library(lme4)
library(lmerTest)
library(tidyverse)

setwd("/home/qy2672/learning-alternations")
model_data <- read.csv("experiments/model_data.csv")
model_data_balanced <- model_data %>%
  filter(is_balanced == 0.5)
model_data_default <- model_data %>%
  filter(is_balanced == -0.5)

# full model: all interaction terms with balance
m_full <- lmer(
  score ~ length_difference + recipient_pronoun + theme_pronoun + (1 | verb_lemma),
  data = model_data_balanced,
  REML = FALSE
)

# remove balance interaction with length
m_length <- lmer(
  score ~ recipient_pronoun + theme_pronoun + (1 | verb_lemma),
  data = model_data_balanced,
  REML = FALSE
)

# full model: all interaction terms with balance
m_full_d <- lmer(
  score ~ length_difference + recipient_pronoun + theme_pronoun + (1 | verb_lemma),
  data = model_data_default,
  REML = FALSE
)

# remove balance interaction with length
m_length_d <- lmer(
  score ~ recipient_pronoun + theme_pronoun + (1 | verb_lemma),
  data = model_data_default,
  REML = FALSE
)

# # remove balance interaction with recipient pronoun
# m_recpron <- lmer(
#   score ~ length_difference + recipient_pronoun + theme_pronoun + is_balanced + 
#     length_difference:is_balanced + theme_pronoun:is_balanced + (1 | verb_lemma),
#   data = model_data,
#   REML = FALSE
# )

# # remove balance interaction with theme pronoun
# m_thmpron <- lmer(
#   score ~ length_difference + recipient_pronoun + theme_pronoun + is_balanced + 
#     length_difference:is_balanced + recipient_pronoun:is_balanced + (1 | verb_lemma),
#   data = model_data,
#   REML = FALSE
# )

print(summary(m_full))
print(summary(m_full_d))

# print(anova(m_length, m_full, test = "LRT"))
# print(anova(m_length_d, m_full_d, test = "LRT"))

# print(anova(m_recpron, m_full, test = "LRT"))
# print(anova(m_thmpron, m_full, test = "LRT"))
