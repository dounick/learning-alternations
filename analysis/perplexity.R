models <- c("Default", "Balanced", "No Datives", "No Ditransitives", "Swapped Datives", "Short-first", "Random-first", "Long-first", "Long-first headfinal")
perplexity1 <- c(55.8783, 56.7330, 54.7594, 52.5822, 56.8073, 58.7264, 68.9270, 65.0329, 84.4295)
perplexity2 <- c(56.0843, 56.2217, 54.6205, 53.3849, 56.2284, 59.4645, 69.7249, 65.7562, 83.4488)
perplexity3 <- c(56.0703, 56.4282, 54.9127, 52.9432, 55.8687, 59.0923, 70.0242, 65.3294, 84.4588)
length_correlation1 <- c(-0.42, -0.33, -0.25, -0.22, -0.02, -0.26, -0.16, -0.08, 0.09)
length_correlation2 <- c(-0.44, -0.33, -0.24, -0.22, 0.03, -0.23, -0.14, -0.05, 0.06)
length_correlation3 <- c(-0.43, -0.35, -0.22, -0.22, -0.02, -0.21, -0.12, -0.04, 0.13)

model_data_long <- data.frame(
  Model = rep(models, 3),
  Perplexity = c(perplexity1, perplexity2, perplexity3),
  LengthCorrelation = c(length_correlation1, length_correlation2, length_correlation3),
  Seed = rep(c("Seed 1", "Seed 2", "Seed 3"), each = length(models))
)

seed_pairs <- rbind(
  data.frame(
    Model = models,
    x = perplexity1, 
    y = length_correlation1,
    xend = perplexity2, 
    yend = length_correlation2
  ),
  data.frame(
    Model = models,
    x = perplexity2, 
    y = length_correlation2,
    xend = perplexity3, 
    yend = length_correlation3
  ),
  data.frame(
    Model = models,
    x = perplexity3, 
    y = length_correlation3,
    xend = perplexity1, 
    yend = length_correlation1
  )
)

library(ggplot2)
library(ggrepel)
library(dplyr)

model_avg_positions <- model_data_long %>%
  group_by(Model) %>%
  summarize(
    AvgPerplexity = mean(Perplexity),
    AvgLengthCorrelation = mean(LengthCorrelation)
  )

model_colors <- c(
  "Default" = "#1f77b4",         
  "Balanced" = "#8c564b",
  "No Datives" = "#e377c2",
  "No Ditransitives" = "#7f7f7f",
  "Swapped Datives" = "#bcbd22",
  "Short-first" = "#ff7f0e",     
  "Random-first" = "#2ca02c",    
  "Long-first" = "#d62728",     
  "Long-first headfinal" = "#9467bd"        
)

hull_data <- model_data_long %>%
  group_by(Model) %>%
  slice(chull(Perplexity, LengthCorrelation))

set.seed(5)

ggplot() +
  geom_polygon(data = hull_data, 
               aes(x = Perplexity, y = LengthCorrelation, fill = Model),
               alpha = 0.1) +
  geom_segment(data = seed_pairs, 
               aes(x = x, y = y, xend = xend, yend = yend, color = Model),
               alpha = 0.4, linetype = "dotted") +
  geom_point(data = model_data_long, 
             aes(x = Perplexity, y = LengthCorrelation, color = Model),
             size = 3, alpha = 0.8) +
  geom_text_repel(data = model_avg_positions,
                  aes(x = AvgPerplexity, y = AvgLengthCorrelation, label = Model, color = Model),
                  box.padding = 0.7,
                  point.padding = 0.5,
                  segment.curvature = -0.1,
                  segment.ncp = 3,
                  segment.angle = 20,
                  force = 3,
                  size = 4, show.legend = FALSE, family="Inconsolata") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "darkgrey") +
  scale_color_manual(values = model_colors) +
  scale_fill_manual(values = model_colors) +
  scale_x_continuous(limits = c(50, 90)) +
  theme_classic(base_family = "Palatino", base_size = 16) +
  labs(
    x = "Perplexity",
    y = "Length Correlation"
    ) +
  theme(
    # panel.grid.minor = element_blank(),
    plot.title = element_text(hjust = 0.5),
    plot.caption = element_text(hjust = 0),
    legend.position = "Right",
    legend.box = "vertical"
  )

# ggsave("paper/perplexity.pdf", dpi = 300, height = 3.5, width = 7, device = cairo_pdf)
# kanishka recommended size for post processing:
ggsave("paper/perplexity.pdf", dpi=300, height = 5.29, width=8.12, device=cairo_pdf)