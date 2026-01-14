
## Yahoo is no longer giving away historical data for free.
## Jan 13, 2025, combine previous Yahoo data with recent
## data from https://fred.stlouisfed.org/series/SP500
## Updated on Jan 13, 2026

sp1 <- read.table("https://ionides.github.io/531w24/03/sp500.csv",sep=",",header=TRUE) # yahoo

sp2 <- read.table("fred-sp500-new.csv",sep=",",header=TRUE) # fred

sp <- data.frame(
  Date=c(sp1$Date,sp2$observation_date),
  Close=c(sp1$Close,sp2$SP500)
)

## The FRED data has NA for holidays
sp <- sp[!is.na(sp$Close),]

write.table(sp,file="sp500-updated.csv")

# check this is okay, then
# mv sp500-updated.csv sp500.csv

