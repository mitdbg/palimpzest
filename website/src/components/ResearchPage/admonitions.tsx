import Admonition from '@theme/Admonition';

interface AbstractProps {
    icon?: React.ReactNode;
    title?: string;
    children: React.ReactNode;
}

export default function Abstract({ children }: AbstractProps) {
  return (
    <div>
      <Admonition type="tip" icon="💡" title="Abstract">
        <p>{children}</p>
      </Admonition>
    </div>
  );
}
