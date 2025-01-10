Knowledge_Prompt = """
Your task is to generate one corresponding wikipedia document based on the given query to help the LLM answer questions.

Demostration:

Origin Question: How many episodes in a season of vampire diaries?

Query: The Vampire Diaries episode count

Document: The Vampire Diaries has a total of 171 episodes over 8 seasons. The show's first season had 22 episodes, the second season had 22 episodes, the third season had 22 episodes, the fourth season had 23 episodes, the fifth season had 22 episodes, the sixth season had 22 episodes, the seventh season had 22 episodes, and the eighth season had 16 episodes.

###

Origin Question: Who developed an explanation for the photoelectric effect?

Query: Photoelectric Effect Explanation

Document: To make sense of the fact that light can eject electrons even if its intensity is low, Albert Einstein proposed that a beam of light is not a wave propagating through space, but rather a collection of discrete wave packets (photons), each with energy hν. This shed light on Max Planck's previous discovery of the Planck relation (E = hν) linking energy (E) and frequency (ν) as arising from quantization of energy. The factor h is known as the Planck constant. In 1887, Heinrich Hertz discovered that electrodes illuminated with ultraviolet light create electric sparks more easily. In 1900, while studying black-body radiation, the German physicist Max Planck suggested that the energy carried by electromagnetic waves could only be released

###

Origin Question: District of maharashtra that are part of red corridor?

Query: Red Corridor in Maharashtra districts

Document: The Red Corridor in Maharashtra includes the following districts: Chandrapur, Gondia, and Gadchiroli.

###

Origin Question: Who played jason in mighty morphin power rangers?

Query: Mighty Morphin Power Rangers Jason

Document: from Dairanger were featured in the second season while only the Kakuranger mecha was featured in the third season, though the Kakuranger costumes were later used for the mini-series Mighty Morphin Alien Rangers. The series was produced by MMPR Productions and distributed by Saban Entertainment, while the show's merchandise was produced and distributed by Bandai Entertainment. The series was well known for its campy tone. In 2010, a re-version of Mighty Morphin Power Rangers, with a revised new look of the original 1993 logo, comic book-referenced graphics, and extra alternative visual effects, was broadcast on ABC Kids, and Bandai produced brand new toys to coincide with the series. Only the first 32 of season one's 60 episodes were remade.

###

Origin Question: {}

Query: {}

Document: """

