library(tidyverse)
library(ggtext)

rawdata <- read_csv("analysis/all_data.csv")

filter(rawdata, is.na(recipient_pos_spacy)) %>%
  select(sentence, recipient_pos, recipient_pos_spacy)

d =  rawdata %>%
  select(global_idx, recipient_pos, recipient_anim, theme_pos, theme_anim,
         loose_default_ratio:long_first_headfinal_ratio,
         length_difference,
         verb_lemma) %>%
  pivot_longer(cols=loose_default_ratio:long_first_ratio, names_to = "condition", values_to = "score") %>% 
  # pivot_longer(cols = `babylm-default_ratio`:`long_first_nodatives_ratio`, names_to=c("variable")) %>%
  mutate(recipient_pos = ifelse(recipient_pos == "PRON" | is.na(recipient_pos), "pronoun", "NP"),
         theme_pos = ifelse(theme_pos == "PRON" | is.na(theme_pos), "pronoun", "NP")) %>%
  # mutate(condition = gsub("small_ratio", "", condition)) %>%
  mutate(condition = str_replace(condition, "(_small)?_ratio", "")) %>%
  mutate(
    condition = factor(
      condition, 
      levels = c("loose_default", "loose_balanced", "datives_removed", "ditransitives_removed", "counterfactual", "short_first", "random_first", "long_first", "long_first_headfinal"),
      labels = c("Unablated (Loose)", "Balanced (Loose)", "No Datives", "No Ditransitives", "Counterfactual",
                 "Short-first\n(No Ditransitives)", "Random-first\n(No Ditransitives)", "Long-first\n(No Ditransitives)", "Long-first\n(Head Final)")
    )
  )


d %>% count(condition)


# chosen.levels = c("babylm-default",
#                   "babylm-balanced",
#                   "loose-balanced",
#                   "loose-default",        
#                   "ditransitives_removed",
#                   "datives_removed",
#                   "short_first_nodatives",
#                   "long_first_nodatives")

# Kyle's first plot
# slopes = d %>%
#   group_by(condition, recipient_pronoun) %>%
#   summarise(slope = cor(score, length_difference)) %>%
#   mutate(x = -3, y=ifelse(recipient_pronoun == "NP", 1, .5),
#          val = as.character(round(slope, 2)))

# Main exp plots
slopes2 = d %>%
  group_by(condition) %>%
  summarise(slope = cor(score, length_difference)) %>%
  mutate(x = -1.85, y=1.35,
         val = paste("<i>r</i> =",as.character(format(round(slope, 2), nsmall=2))))

# plot2_data = d %>%
#   group_by(length_difference, condition, recipient_pronoun) %>%
#   summarise(m = mean(score), .groups = "drop")
# 
# plot2_data$condition = factor(plot2_data$condition,
#                              levels = unique(d$condition))
# 
# ggplot(plot2_data, aes(x = length_difference, y = m, colour = recipient_pronoun)) +
#   geom_point() +
#   theme_bw(base_size = 14) +
#   facet_wrap(~factor(condition, levels = unique(d$condition)), nrow = 1) +
#   ylab("P(DO alternant) - P(PO alternant)") +
#   geom_smooth(method = "lm") +
#   scale_colour_manual(values = c("black", "darkorange")) +
#   geom_text(data=slopes, 
#             aes(x=x, y=y, label=val), size=3)

# Second plot (without recipient_pronoun as a factor)
plot2_data = d %>%
  group_by(length_difference, condition) %>%
  summarise(m = mean(score), .groups = "drop")

# plot2_data$condition = factor(plot2_data$condition,
#                              levels = unique(d$condition))


plot2_data %>%
  # filter(condition %in% c("Strict (unablated)", "Strict (balanced)", "Short-first (No Datives)", "Long-first (No Datives)")) %>%
  # filter(!condition %in% c("Unablated (Strict)", "Balanced (Strict)")) %>%
  filter(!str_detect(condition, "\\(No Datives\\)")) %>% 
  ggplot(aes(x = length_difference, y = m)) +
    geom_point(size = 2, alpha = 0.5, color = "black") +
    facet_wrap(~condition, nrow = 1) +
    ylab("P(DO alternant) - P(PO alternant)") +
    xlab("log(len(recipient)) - log(len(theme))") + 
    geom_smooth(method = "lm") +
    # scale_colour_manual(values = c("black", "darkorange")) +
    geom_richtext(
      data=slopes2 %>% filter(!condition %in% c("Unablated (Strict)", "Balanced (Strict)")) %>%
        filter(!str_detect(condition, "\\(No Datives\\)")), 
      aes(x=x, y=y, label=val), size=4, family="CMU Serif", color = "black",
      fill="cornsilk"
    ) +
  theme_bw(base_size = 16, base_family = "Times") +
  theme(
    panel.grid = element_blank(),
    strip.text = element_text(size = 15),
    axis.text = element_text(color = "black")
  )+
    labs(
      y = "DO Preference",
      x = "Log Difference in Recipient and Theme Lengths"
    )

ggsave("paper/length-pref.pdf", dpi=300, height=3.80, width=14.10, device=cairo_pdf)

#################
library(lme4)

# just unablated
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
            (1 + length_difference + animacy_contrast + pos_contrast||verb_lemma),
          REML=F)
  a = anova(l)
  print(a)
 x = rbind(x, c(i, round(fixef(l)[2], 2),
                round(fixef(l)[3], 2),
                round(fixef(l)[4], 2),
                round( a$`Pr(>F)`[1], 10),
                round( a$`Pr(>F)`[2], 10),
                round( a$`Pr(>F)`[3], 10)
  ))
  print(i)
  print(anova(l))
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
          (1 + length_difference + animacy_contrast + pos_contrast ||verb_lemma),
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
                 (1 + length_difference + animacy_contrast + pos_contrast ||verb_lemma),
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
                  (1 + length_difference + animacy_contrast + pos_contrast ||verb_lemma),
                REML=F)
anova(l.gram)
summary(l.gram)



