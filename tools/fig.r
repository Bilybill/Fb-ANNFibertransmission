library(RcppCNPy)

vlim = c(-8, -4)

x = readBin('./tempdata/w1.bin', n = 14400*8464, what = numeric(), size = 4)
x = log10(abs(x))
dim(x) = c(8464, 14400)

y = readBin('./tempdata/w2.bin', n = 14400*8464, what = numeric(), size = 4)
y = log10(abs(y))
dim(y) = c(8464, 14400)

x = as.numeric(x)
y = as.numeric(y)
xkde = density(x, from = vlim[1], to = vlim[2], bw = 0.01)
ykde = density(y, from = vlim[1], to = vlim[2], bw = 0.01)

xkde$y = 100 * xkde$y / sum(xkde$y)
ykde$y = 100 * ykde$y / sum(ykde$y)

ylim = max(c(xkde$y, ykde$y))

png('rimage/kde2.png', width = 1200, height = 800)
plot(
  x = xkde$x, xlim = vlim, xlab = 'log10(x)', xaxs = 'i',
  y = xkde$y, ylim = c(0, ylim), ylab = '',
  type = 'l', col = 'red', lwd = 2, 
  las = 1, family = 'serif', 
  cex.axis = 2.0, cex.lab = 2.5
)
lines(
  x = ykde$x,
  y = ykde$y,
  type = 'l', col = 'blue', lwd = 2
)
abline(v = (-10):0, lwd = 1, lty = 2, col = '#888888')
abline(h = seq(0.0, 3.0, by = 0.5), lwd = 1, lty = 2, col = '#888888')
# legend(
#   "topleft",c('Our method','CP method'),
#   col = c('red','blue')
# )
box(lwd = 2)
dev.off()
