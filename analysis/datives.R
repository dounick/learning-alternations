library(tidyverse)
library(lme4)

# Read and process data
d = read_csv("all_data.csv") %>%
  select(global_idx, recipient_pronoun, theme_pronoun,
         `babylm-default_ratio`:`long_first_nodatives_ratio`,
         length_difference,
         verb_lemma,
         -random_ratio,
         -long_first_ratio,
         -short_first_ratio) %>%
  pivot_longer(cols = `babylm-default_ratio`:`long_first_nodatives_ratio`, names_to=c("variable")) %>%
  mutate(recipient_pronoun = ifelse(recipient_pronoun > 0, "pronoun", "NP")) %>%
  mutate(variable = gsub("_ratio", "", variable))

chosen.levels = c("babylm-default",
                  "babylm-balanced",
                  "loose-balanced",
                  "loose-default",        
                  "ditransitives_removed",
                  "datives_removed",
                  "short_first_nodatives",
                  "long_first_nodatives")
slopes = d %>%
  group_by(variable, recipient_pronoun) %>%
  summarise(slope = cor(value, length_difference)) %>%
  mutate(x = -3, y=ifelse(recipient_pronoun == "NP", 1, .5),
         val = as.character(round(slope, 2)))

slopes2 = d %>%
  group_by(variable) %>%
  summarise(slope = cor(value, length_difference)) %>%
  mutate(x = -3, y=1,
         val = as.character(round(slope, 2)))


# Second plot (with recipient_pronoun as a factor)
plot2_data = d %>%
  group_by(length_difference, variable, recipient_pronoun) %>%
  summarise(m = mean(value), .groups = "drop")

plot2_data$variable = factor(plot2_data$variable,
                                levels = unique(d$variable))

ggplot(plot2_data, aes(x = length_difference, y = m, colour = recipient_pronoun)) +
  geom_point() +
  theme_bw(base_size = 14) +
  facet_wrap(~factor(variable, levels = chosen.levels), nrow = 1) +
  ylab("DO Preference") +
  geom_smooth(method = "lm") +
  scale_colour_manual(values = c("black", "darkorange")) +
  geom_text(data=slopes, 
            aes(x=x, y=y, label=val), size=3)
  

# Second plot (without recipient_pronoun as a factor)
plot2_data = d %>%
  group_by(length_difference, variable) %>%
  summarise(m = mean(value), .groups = "drop")

plot2_data$variable = factor(plot2_data$variable,
                             levels = unique(d$variable))

ggplot(plot2_data, aes(x = length_difference, y = m)) +
  geom_point() +
  theme_bw(base_size = 14) +
  facet_wrap(~factor(variable, levels = chosen.levels), nrow = 1) +
  ylab("DO Preference") +
  geom_smooth(method = "lm") +
  scale_colour_manual(values = c("black", "darkorange")) +
  geom_text(data=slopes2, 
            aes(x=x, y=y, label=val), size=3)

###############
# compare to default

d$variable = factor(d$variable, levels = chosen.levels)
l = lm(data=d,
   value ~ length_difference * variable)
summary(l)

# compare to LONG
d$variable = factor(d$variable, levels = rev(chosen.levels))
l = lm(data=d,
       value ~ length_difference * variable)
summary(l)


