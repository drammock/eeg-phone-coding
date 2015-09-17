#! /usr/bin/env Rscript

## This script reads in CSV files and spits out PDF slides for use as textual
## prompts.

library(stringi)

langs <- c('dutch', 'english', 'hindi', 'hungarian', 'swahili',
           'chinese-simplified', 'chinese-traditional')

fonts <- c(dutch='', english='', hungarian='', swahili='',
           hindi='Devanagari', pinyin='',
           simplified='CJK SC', traditional='CJK TC')
fonts[fonts!=''] <- paste('Noto Sans', fonts[fonts!=''])
fonts[fonts==''] <- 'Noto Sans'

## unicode denormalization
denorm <- function (x) {
  s <- stri_trans_general(x, "Any-NFD")
}

make_slide <- function(i, lang) {
  syl <- denorm(i['syllable'])
  wrd <- denorm(i['word'])
  tra <- denorm(i['translation'])
  hlt <- denorm(i['highlight'])
  if (wrd=='') return()
  par(oma=rep(0, 4), mar=rep(0, 4))
  plot(0, 0, type='n', xlim=c(0, 1), ylim=c(0, 1), axes=FALSE)
  text(0.5, 0.65, syl, col='green', cex=5)
  # change color for part of the word containing the target syllable
  special <- hlt != ''
  bold_part <- ifelse(special, hlt, syl)
  ixs <- stri_locate_first_fixed(wrd, bold_part)
  fst <- stri_sub(wrd, 1, ixs[1] - 1)
  mid <- stri_sub(wrd, ixs[1], ixs[2])
  lst <- stri_sub(wrd, ixs[2] + 1, nchar(wrd))
  text(0.5, 0.35, bquote(.(fst) * phantom(.(mid)) * .(lst)), col='white', cex=3)
  text(0.5, 0.35, bquote(phantom(.(fst)) * .(mid) * phantom(.(lst))), col='green', cex=3)
  if(lang != 'english') {
    text(0.5, 0.2, tra, col='gray', cex=2, family='Noto Sans')
  }
}

for (lang in langs) {
  flang <- ifelse(substr(lang,1,7)=='chinese', substr(lang,9,nchar(lang)), lang)
  fname <- paste0(lang, '.csv')
  df <- read.csv(fname)
  df$highlight[is.na(df$highlight)] <- ''
  cairo_pdf(file.path('slide-prompts', paste0(lang, '.pdf')), width=8, height=4.5,
            onefile=TRUE, family=fonts[flang], bg='black')
  invisible(apply(df, 1, make_slide, lang))
  dev.off()
}
