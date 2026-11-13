# Mixed-model check of host_regression.py (MaAsLin2's mixed form: study as a random effect).
#
#   Rscript host_regression_lme4.R
#
# For each TM7 and its top partner from the fixed-effect model, refit
#   log2 TSS(TM7) ~ log2 TSS(partner) + log2 reads + sub-habitat + (1 | study)
# on the same samples, and compare the partner coefficient with the fixed-effect estimate.
# -> results/host_regression_lme4.tsv
suppressPackageStartupMessages(library(lme4))
here <- tryCatch(dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))),
                 error = function(e) ".")
res_dir <- file.path(here, "results")
d <- read.delim(file.path(res_dir, "host_regression_lme4_input.tsv"))
fx <- read.delim(file.path(res_dir, "host_regression_tm7_hosts.tsv"))
fx <- fx[fx$rank == 1, ]
out <- do.call(rbind, lapply(split(d, d$pair), function(g) {
  pair <- g$pair[1]
  hab <- if (length(unique(g$habitat)) > 1) "+ habitat" else ""
  f <- if (length(unique(g$study)) > 1) as.formula(paste("y ~ x + log2reads", hab, "+ (1 | study)"))
       else as.formula(paste("y ~ x + log2reads", hab))
  m <- tryCatch(if (grepl("study", deparse(f))) lmer(f, data = g, REML = TRUE) else lm(f, data = g),
                error = function(e) NULL)
  if (is.null(m)) return(NULL)
  cf <- summary(m)$coefficients
  ids <- strsplit(pair, "|", fixed = TRUE)[[1]]
  fe <- fx[fx$tm7 == ids[1] & fx$partner == ids[2], ]
  data.frame(tm7 = ids[1], partner = ids[2], n = nrow(g), n_studies = length(unique(g$study)),
             coef_mixed = cf["x", 1], t_mixed = cf["x", 3],
             coef_fixed = if (nrow(fe)) fe$coef_log2 else NA, t_fixed = if (nrow(fe)) fe$t else NA)
}))
write.table(out, file.path(res_dir, "host_regression_lme4.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
cat(sprintf("%d pairs refitted; coefficient correlation fixed vs mixed: %.3f; same sign: %d of %d; mixed t > 2: %d\n",
            nrow(out), cor(out$coef_mixed, out$coef_fixed, use = "complete.obs"),
            sum(sign(out$coef_mixed) == sign(out$coef_fixed), na.rm = TRUE), nrow(out), sum(out$t_mixed > 2)))
