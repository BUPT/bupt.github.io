# blog
Team Blog

## Blog

We are using [GitHub Pages](https://pages.github.com/) to host our blog, powered by [Jekyll] with the nice and flexible two-column Jekyll theme [Minimal Mistakes](https://mmistakes.github.io/minimal-mistakes/).

### Install

```sh
make install
```

### Serve

```sh
make serve
```

### Image Resizing

```sh
# Mac
brew install magicmagick

# Linux
apt install magicmagick

make fit-image
```

### Rules

1. Attachments & Image files: all files need to be saved under the folder `docs/assets/2019/` (2019 is the current year), and you can expect the url of your file is under the url `https://bupt.github.io/awesome-chatbot/assets/2019/`
1. Author Information: all the author information is saved in the file `docs/_data/author.yml`, please free free to add & modify it by yourself. See: <https://mmistakes.github.io/minimal-mistakes/docs/authors/>
1. Add a teasor image with size 500x300 to your post by add the following `YAML Front Matter` to your post:
    ```yaml
    header:
      teaser: /assets/2019/my-awesome-post-teaser-500x300.jpg
    ```

## Maintainers

- @HoneyCatty - Meina SONG, CS Professor, CAD Lab.
- @huan - Huan LI (李卓桓), CS Ph.D., CAD Lab.

## COPYRIGHT & LICENSE

- Code & Docs © 2018-2019 Contributors
- Code released under the Apache-2.0 License
- Docs released under Creative Commons