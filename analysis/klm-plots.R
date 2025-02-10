library(tidyverse)
library(ggtext)

rawdata <- read_csv("analysis/all_data.csv")

d =  rawdata %>%
  select(global_idx, recipient_pronoun, theme_pronoun, recipient_anim,
         datives_removed_ratio:long_first_noditransitive_ratio,
         length_difference,
         verb_lemma) %>%
  pivot_longer(cols=datives_removed_ratio:long_first_noditransitive_ratio, names_to = "condition", values_to = "score") %>% 
  # pivot_longer(cols = `babylm-default_ratio`:`long_first_nodatives_ratio`, names_to=c("variable")) %>%
  mutate(recipient_pronoun = ifelse(recipient_pronoun > 0, "pronoun", "NP")) %>%
  # mutate(condition = gsub("small_ratio", "", condition)) %>%
  mutate(condition = str_replace(condition, "(_small)?_ratio", "")) %>%
  mutate(
    condition = factor(
      condition, 
      levels = c("loose_default", "loose_balanced", "default", "balanced", "datives_removed", "ditransitives_removed", "short_first", "random_first", "long_first", "short_first_noditransitive", "random_first_noditransitive", "long_first_noditransitive"),
      labels = c("Unablated (Loose)", "Balanced (Loose)", "Unablated (Strict)", 
                 "Balanced (Strict)", "No Datives", "No Ditransitives", "Short-first (No Datives)", "Random-first (No Datives)", "Long-first (No Datives)",
                 "Short-first\n(No Ditransitives)", "Random-first\n(No Ditransitives)", "Long-first\n(No Ditransitives)")
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
  filter(!condition %in% c("Unablated (Strict)", "Balanced (Strict)")) %>%
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


animacy_data <- d %>%
  filter(length_difference == 0) %>% 
  # distinct(global_idx, recipient_anim) %>% count(recipient_anim)
  filter(recipient_anim %in% c("a", "i"))

t_tests <- animacy_data %>%
  group_by(condition) %>%
  nest() %>%
  mutate(
    t_test = map_dbl(data, function(x) {
      a = x %>% filter(recipient_anim == "a") %>% pull(score)
      i = x %>% filter(recipient_anim == "i") %>% pull(score)
      
      test = t.test(a, i) %>% 
        broom::tidy() %>%
        pull(p.value)
      
      return(test)
    })
  ) %>%
  select(-data)

anim_pvals <- t_tests %>%
  mutate(
    p_val = case_when(
      t_test <= 0.001 ~ format.pval(t_test, eps = 0.001, scientific=FALSE),
      t_test <= 0.01 ~ format.pval(t_test, eps = 0.01, scientific=FALSE),
      t_test <= 0.05 ~ format.pval(t_test, eps = 0.05, scientific=FALSE),
      TRUE ~ as.character(round(t_test, 3))
    ),
    p_val = case_when(
      !str_starts(p_val, "<") ~ paste("=", p_val),
      TRUE ~ p_val
    ),
    p_val = paste("<i>p</i>", p_val)
  )

a <- animacy_data %>%
  filter(condition == "Balanced (Loose)", recipient_anim == "a") %>%
  pull(score)

i <- animacy_data %>%
  filter(condition == "Balanced (Loose)", recipient_anim == "i") %>%
  pull(score)

t.test(a, i) %>% broom::tidy()

animacy_data %>%
  group_by(recipient_anim, condition) %>%
  summarize(
    ste = 1.96 * plotrix::std.error(score),
    score = mean(score)
  ) %>%
  ungroup() %>%
  mutate(
    recipient_anim = case_when(
      recipient_anim == "a" ~ "Animate",
      TRUE ~ "Inanimate"
    )
  ) %>%
  filter(!str_detect(condition, "-first")) %>%
  filter(!condition %in% c("Unablated (Strict)", "Balanced (Strict)")) %>%
  ggplot(aes(recipient_anim, score)) +
  geom_point(size = 2) +
  geom_linerange(aes(ymin = score-ste, ymax = score+ste)) +
  geom_richtext(x = 1.5, y = -0.05, data = anim_pvals %>% filter(!str_detect(condition, "-first")) %>%
                  filter(!condition %in% c("Unablated (Strict)", "Balanced (Strict)")), 
                aes(label=p_val), family="CMU Serif", fill="cornsilk") +
  scale_y_continuous(limits = c(-0.82, -0.0)) +
  facet_wrap(~condition, nrow=2) +
  theme_bw(base_size = 18, base_family = "Times") +
  theme(
    panel.grid = element_blank(),
    axis.text = element_text(color = "black")
  ) +
  labs(
    x = "Recipient Animacy",
    y = "DO Preference"
  )
  # distinct(global_idx, recipient_anim) %>% 
  # count(recipient_anim)

# ggsave("paper/animacy.pdf", height=3, width=9.5, dpi=300, device=cairo_pdf)
ggsave("paper/animacy.pdf", height=6.42, width=7.85, dpi=300, device=cairo_pdf)

