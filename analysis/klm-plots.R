library(tidyverse)
library(ggtext)

d = read_csv("analysis/all_data.csv") %>%
  select(global_idx, recipient_pronoun, theme_pronoun,
         loose_default_ratio:ditransitives_removed_ratio,
         length_difference,
         verb_lemma) %>%
  pivot_longer(cols=loose_default_ratio:ditransitives_removed_ratio, names_to = "condition", values_to = "score") %>% 
  # pivot_longer(cols = `babylm-default_ratio`:`long_first_nodatives_ratio`, names_to=c("variable")) %>%
  mutate(recipient_pronoun = ifelse(recipient_pronoun > 0, "pronoun", "NP")) %>%
  mutate(condition = gsub("_ratio", "", condition)) %>%
  mutate(
    condition = factor(
      condition, 
      levels = c("loose_default", "loose_balanced", "default", "balanced", "datives_removed", "ditransitives_removed", "short_first_nodatives", "random_first_nodatives", "long_first_nodatives"),
      labels = c("Unablated (Strict)", "Balanced (Loose)", "Unablated (Strict)", 
                 "Balanced (Strict)", "No Datives", "No Ditransitives", "Short-first (No Datives)", "Random-first (No Datives)", "Long-first (No Datives)")
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
slopes = d %>%
  group_by(condition, recipient_pronoun) %>%
  summarise(slope = cor(score, length_difference)) %>%
  mutate(x = -3, y=ifelse(recipient_pronoun == "NP", 1, .5),
         val = as.character(round(slope, 2)))

# Main exp plots
slopes2 = d %>%
  group_by(condition) %>%
  summarise(slope = cor(score, length_difference)) %>%
  mutate(x = -2, y=1.5,
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
  filter(!condition %in% c("Unablated (Loose)", "Balanced (Loose)")) %>%
  ggplot(aes(x = length_difference, y = m)) +
    geom_point(size = 2, alpha = 0.5, color = "black") +
    facet_wrap(~condition, nrow = 1) +
    ylab("P(DO alternant) - P(PO alternant)") +
    xlab("log(len(recipient)) - log(len(theme))") + 
    geom_smooth(method = "lm") +
    # scale_colour_manual(values = c("black", "darkorange")) +
    geom_richtext(
      data=slopes2 %>% filter(!condition %in% c("Unablated (Loose)", "Balanced (Loose)")), 
      aes(x=x, y=y, label=val), size=4, family="CMU Serif", color = "black",
      fill="cornsilk"
    ) +
  theme_bw(base_size = 16, base_family = "Times") +
  theme(
    panel.grid = element_blank(),
    axis.text = element_text(color = "black")
  )+
    labs(
      y = "DO Preference",
      x = "Log Difference in Recipient and Theme Lengths"
    )

ggsave("paper/length-pref.pdf", dpi=300, height=4.34, width=15.20, device=cairo_pdf)

