# Modeling and Analysis of Time Series Data (STATS 531)

## Chapter 13: Simulation of stochastic dynamic models"

As for [Chapter 12](../12), we use material from the short course [Simulation-Based Inference for Epidemiological Dynamics](https://pypomp.github.io/tutorials/sbied/) (SBIED).  We develop a concrete example of the general POMP modeling framework, and we see the theory and practice of implementing a simulator for the model.

The Susceptible-Infected-Recovered (SIR) model used for this chapter is a central concept for epidemiology. For the purposes of STATS 531, we view it as one example of a mechanistic model, which exemplifies a more general process of model development and data analysis. One epidemiological idea used without definition in the lecture is $R_0$, defined to be the expected number of secondary infections arising from one infected individual in a fully susceptible population. The SIR model supposes that previously infected individuals cannot become reinfected, so those in compartment R are protected from infection.

The SIR model in epidemiology is closely related to predator-prey models in ecology. Similar models can be used to describe spread of ideas (including rumors or mis-information) on social networks. 


| Lecture material | Link       |  
|:-----------------|:-----------|
| Slides  |   [pdf](https://pypomp.github.io/tutorials/sbied/chapter1/slides.pdf) |
| Notes  |   [pdf](https://pypomp.github.io/tutorials/sbied/chapter1/notes.pdf) |
| Recording, Summer 2021, Part 1  | [(17 mins)](https://youtu.be/l5YCll5qcP0) | 
| Recording, Summer 2021, Part 2  | [(24 mins)](https://youtu.be/69F4oEjXkug) | 
| Recording, Summer 2021, Part 3  | [(47 mins)](https://youtu.be/XmUQR1Bp8C4) |
| Recording, Summer 2021, Part 4  | [(10 mins)](https://youtu.be/sNcNhvNY2Ro) |



<!--
| Annotated slides | | [pdf](slides-annotated.pdf) |
-->


----------------------

[Back to course homepage](../index.html)  
[Acknowledgements](../acknowledge.html)  
[Source code for these notes](http://github.com/pypomp/tutorials/tree/master/sbied/chapter2)


----------------------
