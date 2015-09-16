#! /usr/bin/env Rscript

## This script reads in CSV files and spits out PDF slides for use as textual
## prompts.

langs <- c('dutch', 'english', 'hindi', 'hungarian', 'swahili',
           'chinese-simplified', 'chinese-traditional')

fonts <- c(dutch='', english='', hungarian='', swahili='',
           hindi='Devanagari', pinyin='',
           simplified='CJK SC', traditional='CJK TC')
fonts[fonts!=''] <- paste('Noto Sans', fonts[fonts!=''])
fonts[fonts==''] <- 'Noto Sans'

## unicode denormalization
denorm <- function (x) {
  s <- stringi::stri_trans_general(x, "Any-NFD")
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
  bold_part <- ifelse(is.na(hlt), syl, hlt)
  ixs <- stringi::stri_locate_first_fixed(wrd, bold_part)
  wrd1 <- stringi::stri_sub(wrd, 1, ixs[1] - 1)
  wrd2 <- stringi::stri_sub(wrd, ixs[1], ixs[2])
  wrd3 <- stringi::stri_sub(wrd, ixs[2] + 1, nchar(wrd))
  #print(c(wrd, wrd1, wrd2, wrd3))
  text(0.5, 0.35, bquote(.(wrd1) * phantom(.(wrd2)) * .(wrd3)), col='white', cex=3)
  text(0.5, 0.35, bquote(phantom(.(wrd1)) * .(wrd2) * phantom(.(wrd3))), col='green', cex=3)
  if(lang != 'english') {
    text(0.5, 0.2, tra, col='gray', cex=2, family='Noto Sans')
  }
}

for (lang in langs) {
  flang <- ifelse(substr(lang,1,7)=='chinese', substr(lang,9,nchar(lang)), lang)
  fname <- paste0(lang, '.csv')
  df <- read.csv(fname)
  cairo_pdf(paste0(lang, '.pdf'), onefile=TRUE, width=8, height=4.5,
            family=fonts[flang], bg='black')
  invisible(apply(df, 1, make_slide, lang))
  dev.off()
}
