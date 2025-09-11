import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Easy to Use',
    Svg: require('@site/static/img/pz-orange-transparent.svg').default,
    description: (
      <>
        Install Palimpzest with pip and get started in minutes.
        Follow our quickstart and join the Discord community to
        get help with your use case.
      </>
    ),
  },
  {
    title: 'Multi-Modal Joins, Maps, and Filters',
    Svg: require('@site/static/img/multimodal.svg').default,
    description: (
      <>
        Join any combination of text, images, audio, and tables.
        Palimpzest also supports maps and filters over any combination of modalities.
      </>
    ),
  },
  {
    title: 'Highly Optimizable',
    Svg: require('@site/static/img/orange-abacus.svg').default,
    description: (
      <>
        Palimpzest's optimizer can leverage labels and/or an LLM judge to
        produce the best implementation of your data processing pipeline.
      </>
    ),
  },
];

function Feature({title, Svg, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
