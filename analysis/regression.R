library(tidyverse)
library(ggtext)

rawdata <- read_csv("analysis/all_data.csv")

filter(rawdata, is.na(recipient_pos_spacy)) %>%
  select(sentence, recipient_pos, recipient_pos_spacy)

d <- rawdata %>%
  select(global_idx, recipient_pos, recipient_anim, theme_pos, theme_anim,
         loose_default_ratio63:long_first_headfinal_ratio42,
         length_difference,
         verb_lemma) %>%
  pivot_longer(
    cols = c(
      matches("loose_default_ratio\\d+"),
      matches("loose_balanced_ratio\\d+"),
      matches("datives_removed_ratio\\d+"),
      matches("ditransitives_removed_ratio\\d+"),
      matches("counterfactual_ratio\\d+"),
      matches("short_first_ratio\\d+"),
      matches("random_first_ratio\\d+"),
      matches("long_first_ratio\\d+"),
      matches("long_first_headfinal_ratio\\d+")
    ), 
    names_to = "condition_seed", 
    values_to = "score"
  ) %>%
  mutate(
    seed = as.integer(str_extract(condition_seed, "\\d+")),
    condition = str_replace(condition_seed, "_ratio\\d+", "")
  ) %>%
  mutate(
    recipient_pos = ifelse(recipient_pos == "PRON" | is.na(recipient_pos), "pronoun", "NP"),
    theme_pos = ifelse(theme_pos == "PRON" | is.na(theme_pos), "pronoun", "NP")
  ) %>%
  mutate(
    condition = factor(
      condition, 
      levels = c("loose_default", "loose_balanced", "datives_removed", "ditransitives_removed", "counterfactual", "short_first", "random_first", "long_first", "long_first_headfinal"),
      labels = c("Unablated (Loose)", "Balanced (Loose)", "No Datives", "No Ditransitives", "Swapped Datives",
                 "Short-first\n(No Ditransitives)", "Random-first\n(No Ditransitives)", "Long-first\n(No Ditransitives)", "Long-first\n(Head Final)")
    )
  )

d %>% count(condition)

library(lme4)

d$recipient_animacy = ifelse(d$recipient_anim == "i", -.5, .5)
d$theme_animacy = ifelse(d$theme_anim == "i", -.5, .5)
d$theme_pos = ifelse(d$theme_pos == "pronoun", .5, -.5)
d$recipient_pos = ifelse(d$recipient_pos == "pronoun", .5, -.5)

d$animacy_contrast = ifelse(d$recipient_animacy == d$theme_animacy, 0,
                            ifelse(d$recipient_animacy > d$theme_animacy, 1, -1))

d$pos_contrast = ifelse(d$recipient_pos == d$theme_pos, 0,
                        ifelse(d$recipient_pos > d$theme_pos, 1, -1))
###############################################

x = NULL
for (i in unique(d$condition)) {
  print(i)
  l =lmerTest::lmer(data=filter(d, condition == i),
                    score ~ 
                      length_difference + animacy_contrast + pos_contrast + 
                      (1 + length_difference + animacy_contrast + pos_contrast|verb_lemma) + (1+ length_difference + animacy_contrast + pos_contrast|seed),
                    REML=F)
  a = anova(l)
  x = rbind(x, c(i, round(fixef(l)[2], 2),
                 round(fixef(l)[3], 2),
                 round(fixef(l)[4], 2),
                 round( a$`Pr(>F)`[1], 10),
                 round( a$`Pr(>F)`[2], 10),
                 round( a$`Pr(>F)`[3], 10)
  ))
  print(i)
  print(summary(l))
}
a = data.frame(x) 
names(a) = c("condition", "length.coef", "anim.coef", "pron.coef", "length.p", "anim.p", "pron.p")

a$length.p = as.numeric(a$length.p)
a$anim.p = as.numeric(a$anim.p)
a$pron.p = as.numeric(a$pron.p)
# Function to format coefficients with significance

format_coef <- function(coef, pval) {
  sig <- ifelse(pval < 0.05, "*", "")
  paste0(coef, sig)
}

# Apply formatting
df <- a %>%
  mutate(
    length.coef = mapply(format_coef, length.coef, length.p),
    anim.coef = mapply(format_coef, anim.coef, anim.p),
    pron.coef = mapply(format_coef, pron.coef, pron.p)
  ) %>%
  select(condition, length.coef, anim.coef, pron.coef)  # Drop p-values

# Print as LaTeX table
print(xtable(df, caption = "Regression Coefficients with Significance", align = c("l", "l", "c", "c", "c")),
      include.rownames = FALSE, sanitize.text.function = function(x) x)


#################
compare.remove = filter(d, condition %in% c("Unablated (Strict)", "No Datives", "No Ditransitives"))
compare.remove$condition = as.factor(as.character(compare.remove$condition))
compare.remove$condition = factor(compare.remove$condition, 
                                  levels=c("Unablated (Strict)", "No Datives", "No Ditransitives"))

l.remove =lmerTest::lmer(data=compare.remove,
                         score ~ 
                           length_difference*condition + 
                           animacy_contrast*condition + 
                           pos_contrast*condition + 
                           (1 + length_difference + animacy_contrast + pos_contrast|verb_lemma) + (1+ length_difference + animacy_contrast + pos_contrast|seed),
                         REML=F)
summary(l.remove)
anova(l.remove)

#########################
balance = filter(d, condition %in% c("Unablated (Loose)", "Balanced (Loose)"))
balance$condition = as.factor(as.character(balance$condition))
balance$condition = factor(balance$condition, 
                           levels=c("Unablated (Loose)", "Balanced (Loose)"))

l.balance =lmerTest::lmer(data=balance,
                          score ~ 
                            length_difference*condition + 
                            animacy_contrast*condition + 
                            pos_contrast*condition + 
                            (1 + length_difference + animacy_contrast + pos_contrast|verb_lemma) + (1+ length_difference + animacy_contrast + pos_contrast|seed),
                          REML=F)
summary(l.balance)
anova(l.balance)

#########################################################################
ords = c("Short-first\n(No Ditransitives)",
         "Random-first\n(No Ditransitives)",
         "Long-first\n(No Ditransitives)")
gram = filter(d, condition %in% ords)
gram$condition = as.factor(as.character(gram$condition))

l.gram =lmerTest::lmer(data=gram,
                       score ~ 
                         length_difference*condition + 
                         animacy_contrast*condition + 
                         pos_contrast*condition + 
                         (1 + length_difference + animacy_contrast + pos_contrast|verb_lemma) + (1+ length_difference + animacy_contrast + pos_contrast|seed),
                       REML=F)
anova(l.gram)
summary(l.gram)


