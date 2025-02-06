library(tidyverse)

# Read and process data
d = read_csv("all_data.csv") %>%
  select(global_idx, recipient_pronoun, theme_pronoun,
         `default_ratio`:`balanced_ratio`,
         length_difference) %>%
  pivot_longer(cols = `default_ratio`:`balanced_ratio`, names_to=c("variable")) %>%
  mutate(recipient_pronoun = ifelse(recipient_pronoun > 0, "pronoun", "NP")) %>%
  mutate(variable = gsub("_ratio", "", variable))

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
  facet_wrap(~factor(variable, levels = unique(d$variable)), nrow = 1) +
  ylab("P(DO alternant) - P(PO alternant)") +
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
  facet_wrap(~factor(variable, levels = unique(d$variable)), nrow = 1) +
  ylab("P(DO alternant) - P(PO alternant)") +
  xlab("log(len(recipient)) - log(len(theme))") + 
  geom_smooth(method = "lm") +
  scale_colour_manual(values = c("black", "darkorange")) +
  geom_text(data=slopes2, 
            aes(x=x, y=y, label=val), size=3)




