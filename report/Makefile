report.pdf: report.md
	pandoc report.md -o report.pdf \
	    --pdf-engine=xelatex \
	    -V CJKmainfont="WenQuanYi Zen Hei" \
	    -V monofont="DejaVu Sans Mono" \
	    --highlight-style=tango \
	    --toc \
	    -V geometry:margin=2.5cm

clean:
	rm -f report.pdf